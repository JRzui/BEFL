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
	model             *python3.PyObject //FL task model
	dataset           *python3.PyObject
	model_size        [][]int
	Round             int //the current training round
	LocalModelUpdates [][]float64
}

/*
	Returns: (clients, test_dataset, unlabeled_dataset, task_model)
*/
func CreateClients(num int) ([]*Client, *python3.PyObject, *python3.PyObject, *python3.PyObject, [][]int, [][]float64) {
	log.Println("Acquring python lock...")
	runtime.LockOSThread()
	gstate := python3.PyGILState_Ensure()
	res := gopy.Init.CallFunctionObjArgs(gopy.ArgFromString(Task), gopy.ArgFromInt(num), gopy.ArgFromFloat(P), gopy.ArgFromFloat(Lr)) //should get a 4 elements tuple

	dataFeeders := python3.PyTuple_GetItem(res, 0)
	testData := python3.PyTuple_GetItem(res, 1)
	unlabeledData := python3.PyTuple_GetItem(res, 2)
	model := python3.PyTuple_GetItem(res, 3)
	size := python3.PyTuple_GetItem(res, 4)
	model_size := gopy.PyListTuple_Int(size)
	param := python3.PyTuple_GetItem(res, 5)
	globalParam := gopy.PyListList_Float(param)

	clients := make([]*Client, 0)
	for i := 0; i < num; i++ {
		dataFeeder := python3.PyList_GetItem(dataFeeders, i)
		client := &Client{
			ID:         i,
			dataset:    dataFeeder,
			model:      model,
			model_size: model_size,
			Round:      0,
		}
		clients = append(clients, client)
	}

	python3.PyGILState_Release(gstate)
	log.Println("Released python lock.")
	//runtime.UnlockOSThread()
	return clients, testData, unlabeledData, model, model_size, globalParam
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
	deltaPy := gopy.Client_run.CallFunctionObjArgs(c.model, c.dataset, round_model, gopy.ArgFromListArray_Int(c.model_size), gopy.ArgFromInt(K), gopy.ArgFromInt(B))
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
	if chain.ValidLocalTx(tx, nodes[id].Task.ModelSize) {
		nodes[id].Blockchain.AddTransaction(tx)
		nodes[id].SendTx(tx, conn)
	}

}

/*Client 'download' global model from a nearest node (randomly picked)*/
func (c *Client) GetGlobalModel(nodes []*node.Node) ([][]float64, int) {
	id := chain.Random(0, len(nodes)) //pick random nodes

	globalModel, round := nodes[id].GetGlobalParam(nodes[id].Task.TaskName) //get global model from node id
	for {
		//check if the global model is updated and is not null
		if round >= c.Round && (!reflect.DeepEqual(globalModel, [][]float64{})) {
			break
		}
		id = chain.Random(0, len(nodes)) //pick random nodes
		globalModel, round = nodes[id].GetGlobalParam(nodes[id].Task.TaskName)
	}

	return globalModel, round
}
