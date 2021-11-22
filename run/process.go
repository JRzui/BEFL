package run

import (
	"encoding/csv"
	"fmt"
	"log"
	"net/rpc"
	"reflect"
	"runtime"

	"github.com/DataDog/go-python3"
	"github.com/JRzui/BCFedMI/chain"
	"github.com/JRzui/BCFedMI/client"
	"github.com/JRzui/BCFedMI/consensus"
	"github.com/JRzui/BCFedMI/gopy"
	"github.com/JRzui/BCFedMI/network"
	"github.com/JRzui/BCFedMI/node"
)

//New FL task
func NewTask(model *python3.PyObject, unlabel *python3.PyObject, size [][]int, comp_size []int, globalParam [][]float64, momentum [][]float64, rank int, beta float64, slr float64) network.TaskInfo {

	task := network.TaskInfo{
		TaskName:      client.Task,
		Model:         model,
		GlobalParam:   globalParam,
		Momentum:      momentum,
		ModelSize:     size,
		CompModelSize: comp_size,
		Rank:          rank,
		Beta:          beta,
		Slr:           slr,
		UnlabeledData: unlabel,
	}

	return task
}

//Task publish
func TaskPublish(task network.TaskInfo, bcnet *network.BlockchainNetwork) {
	//send task to the network

	//the rpc call cannot be used since the pointer type of Py_Object cannot be transmited
	//error message: "gob: type python3.PyObject has no exported fields"
	//pseudo-publish instead

	bcnet.Task = task
	bcnet.NewTask = true
	log.Println("New task published")
}

//Nodes get task from network
func NodesGetTask(nodes []*node.Node, bcnet *network.BlockchainNetwork) {
	//the rpc call cannot be used since the pointer type of Py_Object cannot be transmitted
	//error message: "gob: type python3.PyObject has no exported fields"
	//pseudo-get instead
	for _, node := range nodes {
		_, found := bcnet.Nodes.Set[node.ID]
		if found == false {
			log.Printf("Node %d has not registered for the network.\n", node.ID)
			return
		}
		node.Task = bcnet.Task
		log.Printf("node %d gets new task info from network.\n", node.ID)
	}
}

//Block preparation
func ProcessBlockPre(nodes []*node.Node, round int, conn *rpc.Client) {
	//candidate block generation
	randIdxs := chain.RandomArray(len(nodes), 0, len(nodes))
	for _, i := range randIdxs {
		nodes[i].GetTxs(conn) //get pending transactions from network

		//check if enough updates are collected
		if len(nodes[i].Blockchain.Transactions.Keys()) >= client.M {
			//generate candidate block
			fmt.Println("start generating candidate block")
			deltas := make([]chain.LocalTransaction, 0)
			for _, key := range nodes[i].Blockchain.Transactions.Keys() {
				deltas = append(deltas, nodes[i].Blockchain.Transactions.Transactions[key])
			}

			globalParam, momentum, _ := nodes[i].GetGlobalParam(nodes[i].Task.TaskName)
			tx := chain.CreateTx(client.Task, round, deltas, globalParam, momentum, nodes[i].Task.Beta, nodes[i].Task.Slr,
				nodes[i].Task.Rank, nodes[i].Task.ModelSize, nodes[i].Task.UnlabeledData, nodes[i].Task.Model)
			candBlock := chain.CreateBlock(nodes[i].Blockchain.LastBlock(), []chain.Transaction{tx})
			//send candidate block to the network
			var sent bool
			err := conn.Call("BlockchainNetwork.SendBlock", candBlock, &sent)
			if err != nil {
				fmt.Println(err)
			}
			if sent {
				fmt.Println("Candidate block generated and sent to verification")
				log.Printf("Node %d sent candidate block to the network\n", nodes[i].ID)
			}
			break
		}
	}
}

//Constitute the committee
func ProcessCommittee(nodes []*node.Node, conn *rpc.Client) {
	for _, node := range nodes {
		var committeeSetup bool

		check, vrfV, vrfP := node.Vrf.GetRole(node.Blockchain.LastBlock())
		if check {
			member := chain.Member{
				ID:       node.ID,
				Address:  node.Address,
				Pk:       node.Vrf.RolesPk,
				VrfValue: vrfV,
				VrfProof: vrfP,
			}
			conn.Call("BlockchainNetwork.SendRole", member, &committeeSetup)
		}

		if committeeSetup {
			log.Println("Committee setup.")
			break
		}

	}
}

//Update the committee info from the blockchain network
func NodesCommitteeUpdate(nodes []*node.Node, conn *rpc.Client) {
	for _, node := range nodes {
		var members []chain.Member
		conn.Call("BlockchainNetwork.CommitteeUpdate", node.ID, &members)
		for _, member := range members {
			node.Blockchain.Committee.Add(member.ID, member)
		}
		log.Printf("node %d committee info updated\n", node.ID)
	}
}

//Achieve the consensus via byzantine fault tolerance
func ProcessBlock(bcnet *network.BlockchainNetwork, nodes []*node.Node, testData *python3.PyObject, conn *rpc.Client, w *csv.Writer) {
	bcnet.VoteLock.Lock()
	//the voting part
	for _, candidate := range bcnet.CandidateBlock {
		block := consensus.Vote(candidate, bcnet.Members)
		if reflect.DeepEqual(block, chain.Block{}) {
			log.Println("Candidate block is not valid, move to next candidate block validation.")
		} else {
			bcnet.VerifiedBlock = block
			bcnet.NewBlock = true
			bcnet.Members = nil        // clear current committee, waiting for the next round committee constitution
			bcnet.Transactions = nil   // clear pending transactions in the current training round
			bcnet.CandidateBlock = nil // clear candidate block pool
			bcnet.BlockReceived = false
			bcnet.CommitteeSetup = false
			log.Println("New block generated.")

			//model test
			globalModel := gopy.ArgFromListArray_Float(bcnet.VerifiedBlock.Transactions[0].GlobalModel)
			acc := Test(bcnet.Task.Model, globalModel, testData, bcnet.Task.ModelSize)
			w.Write([]string{fmt.Sprintf("%v", acc)})
			fmt.Println("New block generated, Test accuracy: ", acc)
			break
		}
	}
	bcnet.VoteLock.Unlock()
}

func ProcessNextRound(nodes []*node.Node, conn *rpc.Client) {
	for _, node := range nodes {
		var newBlock chain.Block
		conn.Call("BlockchainNetwork.GetBlock", node.ID, &newBlock)
		check := chain.ValidVerifiedBlock(node.Blockchain.LastBlock(), newBlock)
		if check {
			node.Blockchain.AddBlock(newBlock)
			log.Printf("Node %d: new block added\n", node.ID)
			node.Blockchain.CommitteeClear()                 //clear this round committee info, wait for next round
			node.Blockchain.Transactions = chain.NewTxPool() //clear transaction pool for current training round
		} else {
			log.Printf("Node %d: the new block published is not valid\n", node.ID)
		}
	}
}

func ProcessFL(workers []*client.Client, attackers []*client.Client, round int, nodes []*node.Node, conn *rpc.Client) {
	num_adver := int(float32(client.M) * float32(client.Cm))
	idxs := chain.RandomArray(client.M-num_adver, 0, len(workers)) //get the random idxs of honest workers in this round

	fmt.Println("FL clients start training...")
	for _, attacker := range attackers {
		round_model, global_r := attacker.GetGlobalModel(nodes)
		attacker.Attack(round_model, global_r, client.K, client.B)
		attacker.SendUpdates(nodes, round, conn)
	}

	for _, idx := range idxs {
		round_model, global_r := workers[idx].GetGlobalModel(nodes)
		workers[idx].Train(round_model, global_r, client.K, client.B)
		workers[idx].SendUpdates(nodes, round, conn)
	}
	fmt.Println("FL clients one round training complete.")
}

func BlockPrint(block chain.Block) {
	fmt.Printf("Index:         %d\n", block.Index)
	fmt.Printf("TimeStamp:     %d\n", block.TimeStamp)
	fmt.Printf("PrevBlockhash: %s\n", block.PrevBlockHash)
	fmt.Println("Transactions:")
	for _, tx := range block.Transactions {
		fmt.Printf("Round %d\n", tx.Round)
		fmt.Println("Global model:")
		fmt.Println(tx.GlobalModel)
	}
	fmt.Println("Signatures:")
	for _, sig := range block.Signatures {
		fmt.Printf("%x\n", sig.Sig)
	}
}

func Test(model *python3.PyObject, globalModel *python3.PyObject, testData *python3.PyObject, model_size [][]int) float64 {
	log.Println("Acquring python lock...")
	runtime.LockOSThread()
	gstate := python3.PyGILState_Ensure()
	accPy := gopy.Test.CallFunctionObjArgs(model, gopy.ArgFromListArray_Int(model_size), globalModel, testData)
	acc := gopy.PyToFloat(accPy)

	python3.PyGILState_Release(gstate)
	log.Println("Released python lock.")
	return acc
}
