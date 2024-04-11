package client

import (
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log"
	"net/rpc"
	"reflect"
	"runtime"

	"github.com/DataDog/go-python3"
	"github.com/JRzui/BEFL/chain"
	"github.com/JRzui/BEFL/gopy"
	"github.com/JRzui/BEFL/node"
)

//FL client
type Client struct {
	ID                int //the id of client
	PublicKey         rsa.PublicKey
	privateKey        *rsa.PrivateKey
	role              *python3.PyObject //FL task model
	model_size        [][]int
	Round             int //the current training round
	LocalModelUpdates [][]float64
}

/*
	Returns: (clients, test_dataset, unlabeled_dataset, task_model)
*/
func CreateClients(attack string) ([]*Client, []*Client, *python3.PyObject, *python3.PyObject, *python3.PyObject, [][]int, []int, [][]float64, [][]float64) {
	log.Println("Acquring python lock...")
	runtime.LockOSThread()
	gstate := python3.PyGILState_Ensure()
	res := gopy.Init.CallFunctionObjArgs(gopy.ArgFromString(Task), gopy.ArgFromFloat(Lr), gopy.ArgFromInt(Rank)) //should get a 4 elements tuple

	dataFeeders := python3.PyTuple_GetItem(res, 0)
	testData := python3.PyTuple_GetItem(res, 1)
	unlabeledData := python3.PyTuple_GetItem(res, 2)
	model := python3.PyTuple_GetItem(res, 3)
	optimizer := python3.PyTuple_GetItem(res, 4)
	size := python3.PyTuple_GetItem(res, 5)
	model_size := gopy.PyListTuple_Int(size)
	comp_size := python3.PyTuple_GetItem(res, 6)
	comp_model_size := gopy.PyListToInt(comp_size)
	param := python3.PyTuple_GetItem(res, 7)
	globalParam := gopy.PyListList_Float(param)
	mmt := python3.PyTuple_GetItem(res, 8)
	momentum := gopy.PyListList_Float(mmt)

	//Attacker initialization
	num_adver := int(float32(M) * float32(Cm)) // the number of adversaries
	attackers := make([]*Client, 0)
	switch attack {
	case "LF":
		//LF attacker initializatopm
		for i := 0; i < num_adver; i++ {
			adver_feeder := python3.PyList_GetItem(dataFeeders, i)
			lf_attacker := gopy.LF.CallFunctionObjArgs(model, optimizer, adver_feeder, gopy.ArgFromString(Task))
			priKey, err := rsa.GenerateKey(rand.Reader, 2048)
			if err != nil {
				panic(err)
			}
			worker := &Client{
				ID:         i,
				role:       lf_attacker,
				PublicKey:  priKey.PublicKey,
				privateKey: priKey,
				model_size: model_size,
				Round:      0,
			}
			attackers = append(attackers, worker)
		}
	case "BF":
		//BF attacker initializatopm
		for i := 0; i < num_adver; i++ {
			adver_feeder := python3.PyList_GetItem(dataFeeders, i)
			bf_attacker := gopy.BF.CallFunctionObjArgs(adver_feeder, model, optimizer)
			priKey, err := rsa.GenerateKey(rand.Reader, 2048)
			if err != nil {
				panic(err)
			}
			worker := &Client{
				ID:         i,
				role:       bf_attacker,
				PublicKey:  priKey.PublicKey,
				privateKey: priKey,
				model_size: model_size,
				Round:      0,
			}
			attackers = append(attackers, worker)
		}
	case "None":
		//no attackers
		num_adver = 0
	default:
		panic("Incorrect attack type")
	}

	honests := make([]*Client, 0)
	for i := num_adver; i < ClientsNum; i++ {
		dataFeeder := python3.PyList_GetItem(dataFeeders, i)
		worker := gopy.Worker.CallFunctionObjArgs(dataFeeder, model, optimizer)
		priKey, err := rsa.GenerateKey(rand.Reader, 2048)
		if err != nil {
			panic(err)
		}
		honest := &Client{
			ID:         i,
			role:       worker,
			PublicKey:  priKey.PublicKey,
			privateKey: priKey,
			model_size: model_size,
			Round:      0,
		}
		honests = append(honests, honest)
	}

	python3.PyGILState_Release(gstate)
	log.Println("Released python lock.")
	return attackers, honests, testData, unlabeledData, model, model_size, comp_model_size, globalParam, momentum
}

/*
Client local training:
Args:
- global_model:	the global params of last round
- K:			the local training step
- B:			the batch size
*/
func (c *Client) Train(global_model [][]float64, global_r int, K int, B int) {
	log.Println("Acquring python lock...")
	runtime.LockOSThread()
	gstate := python3.PyGILState_Ensure()

	round_model := gopy.ArgFromListArray_Float(global_model)
	model_size := gopy.ArgFromListArray_Int(c.model_size)
	rank := gopy.ArgFromInt(Rank)
	k := gopy.ArgFromInt(K)
	b := gopy.ArgFromInt(B)
	deltaPy := gopy.Honest_run.CallFunctionObjArgs(c.role, round_model, model_size, rank, k, b)
	c.LocalModelUpdates = gopy.PyListList_Float(deltaPy)
	c.Round = global_r + 1 //wait for the next round training

	deltaPy.DecRef()
	round_model.DecRef()
	model_size.DecRef()
	rank.DecRef()
	k.DecRef()
	b.DecRef()
	python3.PyGILState_Release(gstate)
	log.Println("Released python lock.")
}

/*
Attacker launch attack:
Args:
- global_model:	the global params of last round
- K:			the local training step
- B:			the batch size
*/
func (c *Client) Attack(global_model [][]float64, global_r int, K int, B int) {
	log.Println("Acquring python lock...")
	runtime.LockOSThread()
	gstate := python3.PyGILState_Ensure()

	round_model := gopy.ArgFromListArray_Float(global_model)
	model_size := gopy.ArgFromListArray_Int(c.model_size)
	rank := gopy.ArgFromInt(Rank)
	k := gopy.ArgFromInt(K)
	b := gopy.ArgFromInt(B)
	deltaPy := gopy.Attacker_run.CallFunctionObjArgs(c.role, round_model, model_size, rank, k, b)
	c.LocalModelUpdates = gopy.PyListList_Float(deltaPy)
	c.Round = global_r + 1 //wait for the next round training

	deltaPy.DecRef()
	round_model.DecRef()
	model_size.DecRef()
	rank.DecRef()
	k.DecRef()
	b.DecRef()
	python3.PyGILState_Release(gstate)
	log.Println("Released python lock.")
}

/*Client 'send' updates to a randomly choosed edge node*/
func (c *Client) SendUpdates(nodes []*node.Node, round int, conn *rpc.Client) {
	sig := c.Sign(c.LocalModelUpdates)
	tx := chain.LocalTransaction{c.ID, c.PublicKey, c.Round, c.LocalModelUpdates, sig}

	id := chain.Random(0, len(nodes)) //pick random nodes

	//node id performs tx validation check
	check := chain.ValidLocalTx(tx, nodes[id].Task.CompModelSize)
	if check {
		nodes[id].Blockchain.AddTransaction(tx)
		nodes[id].SendTx(tx, conn)
	}

}

/*Client 'download' global model from a nearest node (randomly picked)*/
func (c *Client) GetGlobalModel(nodes []*node.Node) ([][]float64, int) {
	id := chain.Random(0, len(nodes)) //pick random node

	globalAddr, round := nodes[id].GetGlobalParam(nodes[id].Task.TaskName) //get global model from node id
	var global chain.Global
	content := chain.DownloadIPFS(globalAddr) //download the global model from IPFS
	json.Unmarshal(content, &global)
	globalModel := global.GlobalModel
	for {
		//check if the global model is updated and is not null
		if round >= c.Round && (!reflect.DeepEqual(globalModel, [][]float64{})) {
			break
		}
		id = chain.Random(0, len(nodes))                                      //pick random nodes
		globalAddr, round = nodes[id].GetGlobalParam(nodes[id].Task.TaskName) //get global model from node id
		var global chain.Global
		content := chain.DownloadIPFS(globalAddr) //download the global model from IPFS
		json.Unmarshal(content, &global)
		globalModel = global.GlobalModel
	}

	return globalModel, round
}

func (c *Client) Sign(modelUpdate [][]float64) []byte {
	msg, err := json.Marshal(modelUpdate)
	if err != nil {
		log.Println("Error encoding modelUpdate to bytes")
		return nil
	}
	hash := sha256.Sum256(msg)
	sig, err := rsa.SignPKCS1v15(rand.Reader, c.privateKey, crypto.SHA256, hash[:])
	if err != nil {
		fmt.Println(err)
		log.Println(fmt.Sprintf("client %d cannot sign msg\n", c.ID))
		return nil
	}
	return sig
}
