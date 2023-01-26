package run

import (
	"crypto/elliptic"
	"encoding/csv"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"log"
	"net/rpc"
	"os"
	"reflect"
	"runtime"
	"time"

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
	globalModel := chain.Global{globalParam, momentum}
	content, err := json.Marshal(globalModel)
	if err != nil {
		panic(err)
	}
	globalAddr := chain.UploadIPFS(content)
	task := network.TaskInfo{
		TaskName:      client.Task,
		Model:         model,
		GlobalModel:   globalAddr,
		ModelSize:     size,
		CompModelSize: comp_size,
		CurrentRound:  0,
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
func ProcessBlockPre(nodes []*node.Node, round int, conn *rpc.Client, bcnet *network.BlockchainNetwork) {
	//candidate block generation
	idx := chain.Random(0, len(nodes))
	if !nodes[idx].AmMember() {
		nodes[idx].GetTxs(conn) //get pending transactions from network

		//check if enough updates are collected
		if len(nodes[idx].Blockchain.Transactions.Keys()) >= client.M {
			//generate candidate block
			fmt.Printf("Node %d start generating candidate block\n", idx)
			deltas := make([]chain.LocalTransaction, 0)
			for _, key := range nodes[idx].Blockchain.Transactions.Keys() {
				deltas = append(deltas, nodes[idx].Blockchain.Transactions.Transactions[key])
			}

			globalAddr, round := nodes[idx].GetGlobalParam(nodes[idx].Task.TaskName) //get global model from node id
			var global chain.Global
			content := chain.DownloadIPFS(globalAddr) //download the global model from IPFS
			json.Unmarshal(content, &global)
			globalParam := global.GlobalModel
			momentum := global.Momentum
			var tx chain.Transaction
			if nodes[idx].Malicious {
				tx = chain.CreateTx_Malicious(nodes[idx].Sig, client.Task, round+1, deltas, globalParam, momentum, nodes[idx].Task.Beta, nodes[idx].Task.Slr,
					nodes[idx].Task.Rank, nodes[idx].Task.ModelSize, nodes[idx].Task.UnlabeledData, nodes[idx].Task.Model)
			} else {
				tx = chain.CreateTx(nodes[idx].Sig, client.Task, round+1, deltas, globalParam, momentum, nodes[idx].Task.Beta, nodes[idx].Task.Slr,
					nodes[idx].Task.Rank, nodes[idx].Task.ModelSize, nodes[idx].Task.UnlabeledData, nodes[idx].Task.Model)
			}
			candBlock := chain.CreateBlock(nodes[idx].Blockchain.LastBlock(), []chain.Transaction{tx})
			//send candidate block to the network
			var sent bool
			//panic: gob: type not registered for interface: elliptic.p256Curve
			gob.Register(elliptic.P256())
			err := conn.Call("BlockchainNetwork.SendBlock", network.Candidate{candBlock, nodes[idx].ID}, &sent)
			if err != nil {
				fmt.Println(err)
			}
			if sent {
				fmt.Println("Candidate block generated and sent to verification")
				log.Printf("Node %d sent candidate block to the network\n", nodes[idx].ID)
			} else {
				go func() { bcnet.CandidateWait <- true }()
			}
		} else {
			go func() { bcnet.CandidateWait <- true }()
		}
	} else {
		go func() { bcnet.CandidateWait <- true }()
	}
}

//Constitute the committee
func ProcessCommittee(nodes []*node.Node, conn *rpc.Client, bcnet *network.BlockchainNetwork) {
	var committeeSetup bool

	fmt.Println("Committee constitution...")
	randIdxs := chain.RandomArray(len(nodes), 0, len(nodes))
	for _, id := range randIdxs {
		check, vrfV, vrfP := nodes[id].Vrf.GetRole(nodes[id].ID, chain.CommitteeSize, bcnet.StakeMap, nodes[id].Blockchain.LastBlock())
		if check {
			member := chain.Member{
				ID:       nodes[id].ID,
				Address:  nodes[id].Address,
				Pk:       nodes[id].Vrf.RolesPk,
				VrfValue: vrfV,
				VrfProof: vrfP,
			}
			conn.Call("BlockchainNetwork.SendRole", member, &committeeSetup)
		}
		if committeeSetup {
			log.Println("Committee setup.")
			fmt.Println("Committee setup.")
			break
		}
	}
}

//Update the committee info from the blockchain network
func NodesCommitteeUpdate(nodes []*node.Node, conn *rpc.Client, bcnet *network.BlockchainNetwork) {
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
func ProcessBlock(bcnet *network.BlockchainNetwork, nodes []*node.Node, testData *python3.PyObject, conn *rpc.Client) {
	execute := make(chan bool)
	go func() {
		execute <- true
	}()
	select {
	case <-execute:
		bcnet.VoteLock.Lock()
		candidate := <-bcnet.CandidateBlock
		fmt.Println("Voting start")
		//the voting part
		members := make([]chain.Member, 0)
		for _, member := range bcnet.Members.Members {
			members = append(members, member)
		}
		block := consensus.Vote(candidate, members)
		if reflect.DeepEqual(block, chain.Block{}) {
			fmt.Println("Failed to achieve consensus on the candidate block")
			log.Println("Candidate block is not valid, move to next candidate block validation.")
			go func() { bcnet.CandidateWait <- true }()
			bcnet.VoteLock.Unlock()
			return
		} else {
			bcnet.VerifiedBlock = block
			go func() { bcnet.NewBlock <- true }()
			network.RewardsDistribution(bcnet) // distribute rewards to committee members and block creator
			log.Println("New block generated.")

			//model test
			var global chain.Global
			content := chain.DownloadIPFS(bcnet.VerifiedBlock.Transactions[0].GlobalModel) //download the global model from IPFS
			json.Unmarshal(content, &global)
			globalModel := gopy.ArgFromListArray_Float(global.GlobalModel)
			acc := Test(bcnet.Task.Model, globalModel, testData, bcnet.Task.ModelSize)

			f, err := os.OpenFile(fmt.Sprintf("results/%s_%s.csv", client.Task, client.Attack), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
			if err != nil {
				panic(err)
			}
			writer := csv.NewWriter(f)
			writer.Write([]string{fmt.Sprintf("%v", acc)})
			writer.Flush()
			f.Close()

			fmt.Println("New block generated, Test accuracy: ", acc)
			bcnet.VoteLock.Unlock()

		}
	case <-time.After(chain.VotingTimeout):
		fmt.Println("Voting time out")
		bcnet.Members = chain.NewMemberSet() // clear current committee, waiting for the next round committee constitution
		go func() { bcnet.CommitteeWait <- true }()
		bcnet.VoteLock.Unlock()
	}
}

func ProcessNextRound(bcnet *network.BlockchainNetwork, nodes []*node.Node, conn *rpc.Client, round *int) {
	bcnet.Members = chain.NewMemberSet() // clear current committee, waiting for the next round committee constitution
	bcnet.Transactions = nil             // clear pending transactions in the current training round
	*round++

	var newBlock chain.Block
	for _, node := range nodes {
		conn.Call("BlockchainNetwork.GetBlock", node.ID, &newBlock)
		check := chain.ValidVerifiedBlock(node.Blockchain.LastBlock(), newBlock)
		if check {
			node.Blockchain.AddBlock(newBlock, node.ID)
			log.Printf("Node %d: new block added\n", node.ID)
			node.Blockchain.CommitteeClear()                 //clear this round committee info, wait for next round
			node.Blockchain.Transactions = chain.NewTxPool() //clear transaction pool for current training round
		} else {
			log.Printf("Node %d: the new block published is not valid\n", node.ID)
		}
	}

	go func() { bcnet.CommitteeWait <- true }()
	go func() { bcnet.NewRound <- true }()
	go func() { bcnet.CandidateWait <- true }()
}

func ProcessFL(workers []*client.Client, attackers []*client.Client, round int, nodes []*node.Node, conn *rpc.Client) {
	num_adver := int(float32(client.M) * float32(client.Cm))
	idxs := chain.RandomArray(client.M-num_adver, 0, len(workers)) //get the random idxs of honest workers in this round

	fmt.Println("FL clients start training...")
	for _, attacker := range attackers {
		round_model, global_r := attacker.GetGlobalModel(nodes)
		attacker.Attack(round_model, global_r, client.K, client.B)
		attacker.SendUpdates(nodes, global_r, conn)
	}

	for _, idx := range idxs {
		round_model, global_r := workers[idx].GetGlobalModel(nodes)
		workers[idx].Train(round_model, global_r, client.K, client.B)
		workers[idx].SendUpdates(nodes, global_r, conn)
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

	accPy.DecRef()
	python3.PyGILState_Release(gstate)
	log.Println("Released python lock.")
	return acc
}

func SaveStakeMap(nodes []*node.Node, stakemap map[int]int) {
	f, err := os.OpenFile("results/stake.csv", os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		panic(err)
	}
	writer := csv.NewWriter(f)
	stakes := make([]string, 0)
	for _, node := range nodes {
		stakes = append(stakes, fmt.Sprintf("%v", stakemap[node.ID]))
	}
	writer.Write(stakes)
	writer.Flush()
	f.Close()
}
