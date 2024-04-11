package chain

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log"
	"runtime"

	"github.com/DataDog/go-python3"
	"github.com/JRzui/BEFL/gopy"
)

/* Custom transaction*/
type Transaction struct {
	Task           string             `json:"task"`
	Round          int                `json:"round"`
	GlobalModel    string             `json:"globalModel"`      //the IPFS address of the global model
	GlobalModelSig Signature          `json:"SenderSignauture"` //the signature of uploaded global
	Deltas         []LocalTransaction `json:"modelUpdates"`     //the deltas will be removed when the confirmed block generated
}

type Global struct {
	GlobalModel [][]float64 `json:"roundModel"`
	Momentum    [][]float64 `json:"momentum"`
}

type LocalTransaction struct {
	ClientID    int           `json:"clientID"`
	PublicKey   rsa.PublicKey `json:"publicKey"`
	Round       int           `json:"round"`
	ModelUpdate [][]float64   `json:"modelUpdate"`
	Signature   []byte        `json:"signature"`
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

func CreateTx(priKey *ecdsa.PrivateKey, task string, round int, txs []LocalTransaction, globalParam [][]float64, momentum [][]float64, beta float64, slr float64, rank int, model_size [][]int, unlabel *python3.PyObject, model *python3.PyObject) Transaction {
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
	momentumPy := gopy.ArgFromListArray_Float(momentum)
	betaPy := gopy.ArgFromFloat(beta)
	slrPy := gopy.ArgFromFloat(slr)
	modelSizePy := gopy.ArgFromListArray_Int(model_size)
	rankPy := gopy.ArgFromInt(rank)

	res := gopy.Node_run.CallFunctionObjArgs(deltas, globalParamPy, momentumPy, betaPy, slrPy, unlabel, model, modelSizePy, rankPy)
	globalParamPy_ := python3.PyTuple_GetItem(res, 0)
	momentumPy_ := python3.PyTuple_GetItem(res, 1)
	globalModel := Global{gopy.PyListList_Float(globalParamPy_), gopy.PyListList_Float(momentumPy_)}
	content, err := json.Marshal(globalModel)
	if err != nil {
		panic(err)
	}
	globalAddr := UploadIPFS(content)
	sig, err := ecdsa.SignASN1(rand.Reader, priKey, []byte(globalAddr))
	if err != nil {
		panic("cannot sign the global model")
	}
	tx := Transaction{
		Task:           task,
		Round:          round,
		GlobalModel:    globalAddr,
		GlobalModelSig: Signature{sig, priKey.PublicKey},
		Deltas:         txs,
	}

	//release memory
	res.DecRef()
	deltas.DecRef()
	globalParamPy.DecRef()
	momentumPy.DecRef()
	betaPy.DecRef()
	slrPy.DecRef()
	modelSizePy.DecRef()
	rankPy.DecRef()
	python3.PyGILState_Release(gstate)
	log.Println("Released python lock.")
	return tx
}

func CreateTx_Malicious(priKey *ecdsa.PrivateKey, task string, round int, txs []LocalTransaction, globalParam [][]float64, momentum [][]float64, beta float64, slr float64, rank int, model_size [][]int, unlabel *python3.PyObject, model *python3.PyObject) Transaction {
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
	momentumPy := gopy.ArgFromListArray_Float(momentum)
	betaPy := gopy.ArgFromFloat(beta)
	slrPy := gopy.ArgFromFloat(slr)
	modelSizePy := gopy.ArgFromListArray_Int(model_size)
	rankPy := gopy.ArgFromInt(rank)

	res := gopy.Malicious_node_run.CallFunctionObjArgs(deltas, globalParamPy, momentumPy, betaPy, slrPy, unlabel, model, modelSizePy, rankPy)
	globalParamPy_ := python3.PyTuple_GetItem(res, 0)
	momentumPy_ := python3.PyTuple_GetItem(res, 1)
	globalModel := Global{gopy.PyListList_Float(globalParamPy_), gopy.PyListList_Float(momentumPy_)}
	content, err := json.Marshal(globalModel)
	if err != nil {
		panic(err)
	}
	globalAddr := UploadIPFS(content)
	sig, err := ecdsa.SignASN1(rand.Reader, priKey, []byte(globalAddr))
	if err != nil {
		panic("cannot sign the global model")
	}
	tx := Transaction{
		Task:           task,
		Round:          round,
		GlobalModel:    globalAddr,
		GlobalModelSig: Signature{sig, priKey.PublicKey},
		Deltas:         txs,
	}

	//release memory
	res.DecRef()
	deltas.DecRef()
	globalParamPy.DecRef()
	momentumPy.DecRef()
	betaPy.DecRef()
	slrPy.DecRef()
	modelSizePy.DecRef()
	rankPy.DecRef()
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

func ValidLocalTx(tx LocalTransaction, modelSize []int) bool {
	//check if the modelUpdate is issued by the client
	msg, err := json.Marshal(tx.ModelUpdate)
	if err != nil {
		log.Println("Error encoding modelUpdate to bytes")
		return false
	}
	hash := sha256.Sum256(msg)
	err = rsa.VerifyPKCS1v15(&tx.PublicKey, crypto.SHA256, hash[:], tx.Signature)
	if err != nil {
		log.Println("Incorrect verification")
		return false
	}

	//check if the shape of the local model updates is in line with the model
	if len(tx.ModelUpdate) != len(modelSize) {
		log.Println("Incorrect model length")
		return false
	}

	for i := 0; i < len(modelSize); i++ {
		//check the parameter number in each layer
		if modelSize[i] != len(tx.ModelUpdate[i]) {
			log.Printf("Incorrect model size in layer %d", i)
			return false
		}
	}

	return true
}
