package client

import (
	"log"
	"net/rpc"
	"reflect"
	"runtime"

	"github.com/DataDog/go-python3"
	"github.com/JRzui/BCFedMI/chain"
	"github.com/JRzui/BCFedMI/gopy"
	"github.com/JRzui/BCFedMI/node"
)

//FL client
type Client struct {
	ID                int               //the id of client
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
			worker := &Client{
				ID:         i,
				role:       lf_attacker,
				model_size: model_size,
				Round:      0,
			}
			attackers = append(attackers, worker)
		}
	case "BF":
		//BF attacker initializatopm
		for i := 0; i < num_adver; i++ {
			adver_feeder := python3.PyList_GetItem(dataFeeders, i)
			bf_attacker := gopy.BF.CallFunctionObjArgs(model, optimizer, adver_feeder)
			worker := &Client{
				ID:         i,
				role:       bf_attacker,
				model_size: model_size,
				Round:      0,
			}
			attackers = append(attackers, worker)
		}
	default:
		panic("Incorrect attack type")
	}

	honests := make([]*Client, 0)
	for i := num_adver; i < ClientsNum; i++ {
		dataFeeder := python3.PyList_GetItem(dataFeeders, i)
		woker := gopy.Worker.CallFunctionObjArgs(dataFeeder, model, optimizer)
		honest := &Client{
			ID:         i,
			role:       woker,
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
	deltaPy := gopy.Honest_run.CallFunctionObjArgs(c.role, round_model, gopy.ArgFromListArray_Int(c.model_size), gopy.ArgFromInt(Rank), gopy.ArgFromInt(K), gopy.ArgFromInt(B))
	c.LocalModelUpdates = gopy.PyListList_Float(deltaPy)
	c.Round = global_r + 1 //wait for the next round training

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
	deltaPy := gopy.Attacker_run.CallFunctionObjArgs(c.role, round_model, gopy.ArgFromListArray_Int(c.model_size), gopy.ArgFromInt(Rank), gopy.ArgFromInt(K), gopy.ArgFromInt(B))
	c.LocalModelUpdates = gopy.PyListList_Float(deltaPy)
	c.Round = global_r + 1 //wait for the next round training

	python3.PyGILState_Release(gstate)
	log.Println("Released python lock.")
}

/*Client 'send' updates to a randomly choosed edge node*/
func (c *Client) SendUpdates(nodes []*node.Node, round int, conn *rpc.Client) {
	delta := chain.LocalTransaction{c.ID, c.Round, c.LocalModelUpdates}

	id := chain.Random(0, len(nodes)) //pick random nodes
	tx := delta
	//node id performs tx validation check
	if chain.ValidLocalTx(tx, nodes[id].Task.CompModelSize) {
		nodes[id].Blockchain.AddTransaction(tx)
		nodes[id].SendTx(tx, conn)
	}

}

/*Client 'download' global model from a nearest node (randomly picked)*/
func (c *Client) GetGlobalModel(nodes []*node.Node) ([][]float64, int) {
	id := chain.Random(0, len(nodes)) //pick random node

	globalModel, _, round := nodes[id].GetGlobalParam(nodes[id].Task.TaskName) //get global model from node id
	for {
		//check if the global model is updated and is not null
		if round >= c.Round && (!reflect.DeepEqual(globalModel, [][]float64{})) {
			break
		}
		id = chain.Random(0, len(nodes)) //pick random nodes
		globalModel, _, round = nodes[id].GetGlobalParam(nodes[id].Task.TaskName)
	}

	return globalModel, round
}
