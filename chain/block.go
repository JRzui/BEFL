package chain

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log"
	"reflect"
	"runtime"
	"time"

	"github.com/DataDog/go-python3"
	"github.com/JRzui/BEFL/gopy"
)

/*Custom block structure*/

type Block struct {
	Index         int64         `json:"index"`        // the index of current block
	TimeStamp     int64         `json:"timeStamp"`    // the creation timestamp
	PrevBlockHash string        `json:"prevHash"`     // the previous block hash
	Transactions  []Transaction `json:"transactions"` // transactions contained in the block
	Signatures    []Signature   `json:"signatures"`
}

type Signature struct {
	Sig []byte
	Pk  ecdsa.PublicKey
}

/*Custom Genesis Block*/
func Genesis() Block {
	genesis := Block{
		Index:         0,
		TimeStamp:     00000000,
		PrevBlockHash: "FL custom Blockchain system",
	}
	return genesis
}

func CreateBlock(lastBlock Block, txs []Transaction) Block {
	height := lastBlock.Index + 1
	prevHash := fmt.Sprintf("%x", ComputeHashForBlock(BlockToByteWithSig(lastBlock)))
	block := Block{
		Index:         height,
		TimeStamp:     time.Now().Unix(),
		PrevBlockHash: prevHash,
		Transactions:  txs,
	}
	return block
}

func ValidCandidateBlock(lastBlock Block, candidateBlock Block, globalParam [][]float64, momentum [][]float64, modelSize [][]int, compSize []int, PartNum int, rank int, beta float64, slr float64, unlabel *python3.PyObject, model *python3.PyObject) bool {
	//Check if the previous hash is correct
	blockBytes := BlockToByteWithSig(lastBlock)
	lastHash := ComputeHashForBlock(blockBytes)
	if fmt.Sprintf("%x", lastHash) != candidateBlock.PrevBlockHash {
		log.Println("Incorrect previous hash")
		return false
	}
	if candidateBlock.Index != lastBlock.Index+1 {
		return false
	}

	//Check if the IPFS address contained in the transaction is correct
	if !ecdsa.VerifyASN1(&candidateBlock.Transactions[0].GlobalModelSig.Pk, []byte(candidateBlock.Transactions[0].GlobalModel), candidateBlock.Transactions[0].GlobalModelSig.Sig) {
		log.Println("Incorrect signature of IPFS address")
		return false
	}

	//Check if transactions contained in the block are valid
	for _, tx := range candidateBlock.Transactions[0].Deltas {
		if ValidLocalTx(tx, compSize) == false {
			log.Println("Incorrect wrapped pending transactions")
			return false
		}
	}

	//check if enough updates are collected
	updates_num := len(candidateBlock.Transactions[0].Deltas)
	if updates_num < PartNum {
		log.Println("Not enough model updates for aggregation")
		return false
	}

	//Check if the aggregated global model param is correct
	log.Println("Acquring python lock...")
	runtime.LockOSThread()
	gstate := python3.PyGILState_Ensure() //prevent python stick
	deltas := python3.PyList_New(updates_num)
	for i, tx := range candidateBlock.Transactions[0].Deltas {
		res := python3.PyList_SetItem(deltas, i, gopy.ArgFromListArray_Float(tx.ModelUpdate))
		if res != 0 {
			log.Println("Arg client_grads pyList error")
		}
	}

	globalParamPy := gopy.ArgFromListArray_Float(globalParam)
	momentumPy := gopy.ArgFromListArray_Float(momentum)
	betaPy := gopy.ArgFromFloat(beta)
	slrPy := gopy.ArgFromFloat(slr)
	modelSizePy := gopy.ArgFromListArray_Int(modelSize)
	rankPy := gopy.ArgFromInt(rank)
	res := gopy.Node_run.CallFunctionObjArgs(deltas, globalParamPy, momentumPy, betaPy, slrPy, unlabel, model, modelSizePy, rankPy)
	globalParamPy_ := python3.PyTuple_GetItem(res, 0)
	momentumPy_ := python3.PyTuple_GetItem(res, 1)

	//check if the aggregated global param is correct
	var global Global
	content := DownloadIPFS(candidateBlock.Transactions[0].GlobalModel) //download the global model from IPFS
	json.Unmarshal(content, &global)
	candGlobal := global.GlobalModel
	candMmt := global.Momentum
	globalParam = gopy.PyListList_Float(globalParamPy_) //global param in current training round
	momentum = gopy.PyListList_Float(momentumPy_)

	comp := reflect.DeepEqual(candGlobal, globalParam) && reflect.DeepEqual(candMmt, momentum)
	if !comp {
		log.Println("Incorrect aggregation")
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
	return comp
}

func ValidVerifiedBlock(lastBlock Block, verifiedBlock Block) bool {
	//Check if the previous hash is correct
	blockBytes := BlockToByteWithSig(lastBlock)
	lastHash := ComputeHashForBlock(blockBytes)
	if fmt.Sprintf("%x", lastHash) != verifiedBlock.PrevBlockHash {
		log.Println("Incorrect hash.")
		return false
	}
	if verifiedBlock.Index != lastBlock.Index+1 {
		log.Println("Incorrect block height.")
		return false
	}

	//Check the signatures part
	if len(verifiedBlock.Signatures) <= 2/3*CommitteeSize {
		log.Println("The block didn't get the approval of the committee")
		return false
	}
	yesVoteNum := 0
	for _, sig := range verifiedBlock.Signatures {
		if ecdsa.VerifyASN1(&sig.Pk, BlockToByteWithoutSig(verifiedBlock), sig.Sig) {
			yesVoteNum += 1
		}
	}
	if yesVoteNum <= 2/3*CommitteeSize {
		log.Println("Insufficient committee aprrovals.")
		return false
	}

	return true
}

func BlockToByteWithSig(block Block) []byte {
	blockBytes, _ := json.Marshal(block)
	return blockBytes
}

func BlockToByteWithoutSig(block Block) []byte {
	indexB := Int64ToBytes(block.Index)
	tsB := Int64ToBytes(block.TimeStamp)
	prehashB := []byte(block.PrevBlockHash)
	var txsB []byte
	for _, tx := range block.Transactions {
		txsB = bytes.Join([][]byte{txsB, TxToByteWithoutDeltas(tx)}, []byte{})
	}
	return bytes.Join([][]byte{indexB, tsB, prehashB, txsB}, []byte{})
}

func ComputeHashForBlock(blockBytes []byte) []byte {
	hash := sha256.Sum256(blockBytes)
	return hash[:]
}
