package chain

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log"
	"runtime"

	"github.com/DataDog/go-python3"
	"github.com/JRzui/BCFedMI/gopy"
)

/* Custom transaction*/
type Transaction struct {
	Task        string             `json:"task"`
	Round       int                `json:"round"`
	GlobalModel [][]float64        `json:"globalModel"`
	Deltas      []LocalTransaction `json:"modelUpdates"`
}

type LocalTransaction struct {
	ClientID    int         `json:"clientID"`
	Round       int         `json:"round"`
	ModelUpdate [][]float64 `json:"modelUpdate"`
	//client publick key
	//client signature
}

type TransactionPool struct {
	set          map[string]bool
	Transactions map[string]LocalTransaction
}

func NewTxPool() TransactionPool {
	return TransactionPool{make(map[string]bool), make(map[string]LocalTransaction)}
}

func (txp *TransactionPool) Add(tx LocalTransaction) bool {
	//get the hash string of the tx
	txHash := LocalTxHash(tx)
	_, found := txp.set[txHash] // the found should return false
	txp.set[txHash] = true
	txp.Transactions[txHash] = tx
	return !found
}

func (txp *TransactionPool) Keys() []string {
	var keys []string
	for k, _ := range txp.set {
		keys = append(keys, k)
	}
	return keys
}

func CreateTx(task string, round int, txs []LocalTransaction, globalParam [][]float64, model_size [][]int, unlabel *python3.PyObject, model *python3.PyObject) Transaction {
	log.Println("Acquring python lock...")
	runtime.LockOSThread()
	gstate := python3.PyGILState_Ensure()
	deltas := python3.PyList_New(len(txs))
	for i, tx := range txs {
		res := python3.PyList_SetItem(deltas, i, gopy.ArgFromListArray_Float(tx.ModelUpdate))
		if res != 0 {
			log.Println("Arg client_grads pyList error")
			return Transaction{}
		}
	}

	globalParamPy := gopy.ArgFromListArray_Float(globalParam)
	globalModel := gopy.Node_agg.CallFunctionObjArgs(deltas, globalParamPy, unlabel, model, gopy.ArgFromListArray_Int(model_size))
	tx := Transaction{
		Task:        task,
		Round:       round,
		GlobalModel: gopy.PyListList_Float(globalModel),
		Deltas:      txs,
	}

	python3.PyGILState_Release(gstate)
	log.Println("Released python lock.")
	return tx
}

func TxToByte(tx Transaction) []byte {
	txByte, _ := json.Marshal(tx)
	return txByte
}

func TxToByteWithoutDeltas(tx Transaction) []byte {
	tx.Deltas = nil
	txByte, _ := json.Marshal(tx)
	return txByte
}

func LocalTxToByte(tx LocalTransaction) []byte {
	txByte, _ := json.Marshal(tx)
	return txByte
}

func LocalTxHash(tx LocalTransaction) string {
	txByte := LocalTxToByte(tx)
	hashByte := sha256.Sum256(txByte)
	return fmt.Sprintf("%x", hashByte[:])
}

func ValidLocalTx(tx LocalTransaction, modelSize [][]int) bool {
	//check if the shape of the local model updates is in line with the model
	if len(tx.ModelUpdate) != len(modelSize) {
		return false
	}

	for i := 0; i < len(modelSize); i++ {
		para_num := 1
		for j := 0; j < len(modelSize[i]); j++ {
			para_num *= modelSize[i][j]
		}

		//check the parameter number in each layer
		if para_num != len(tx.ModelUpdate[i]) {
			return false
		}
	}

	return true
}
