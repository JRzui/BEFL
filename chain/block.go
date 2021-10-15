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
	"github.com/JRzui/BCFedMI/gopy"
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

func ValidCandidateBlock(lastBlock Block, candidateBlock Block, globalParam [][]float64, modelSize [][]int, PartNum int, unlabel *python3.PyObject, model *python3.PyObject) bool {
	//Check if the previous hash is correct
	blockBytes := BlockToByteWithSig(lastBlock)
	lastHash := ComputeHashForBlock(blockBytes)
	if fmt.Sprintf("%x", lastHash) != candidateBlock.PrevBlockHash {
		return false
	}
	if candidateBlock.Index != lastBlock.Index+1 {
		return false
	}

	//Check if transactions contained in the block are valid
	for _, tx := range candidateBlock.Transactions[0].Deltas {
		if ValidLocalTx(tx, modelSize) == false {
			return false
		}
	}

	//check if enough updates are collected
	updates_num := len(candidateBlock.Transactions[0].Deltas)
	if updates_num < PartNum {
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

	globalParamPy := gopy.Node_agg.CallFunctionObjArgs(deltas, gopy.ArgFromListArray_Float(globalParam), unlabel, model, gopy.ArgFromListArray_Int(modelSize))
	//check if the aggregated global param is correct
	candGlobal := candidateBlock.Transactions[0].GlobalModel
	globalParam = gopy.PyListList_Float(globalParamPy) //global param in current training round
	comp := reflect.DeepEqual(candGlobal, globalParam)

	python3.PyGILState_Release(gstate)
	log.Println("Released python lock.")
	//runtime.UnlockOSThread()
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
		log.Println("Insufficient committee aprovals.")
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
