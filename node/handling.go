package node

import (
	"bufio"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"encoding/gob"
	"encoding/json"
	"log"
	"net"

	"github.com/JRzui/BEFL/chain"
)

func (n *Node) handleVote(rw *bufio.ReadWriter, Raddr net.Addr) {
	log.Printf("Node %d: Receiving candidate block.\n", n.ID)
	var data chain.Block
	dec := gob.NewDecoder(rw)
	err := dec.Decode(&data)
	if err != nil {
		log.Printf("Node %d: Error decoding the candidate block.\n", n.ID)
		rw.WriteString("N\n") //send No vote back
		rw.Flush()
		return
	}

	globalAddr, _ := n.GetGlobalParam(n.Task.TaskName)
	var global chain.Global
	content := chain.DownloadIPFS(globalAddr) //download the global model from IPFS
	json.Unmarshal(content, &global)
	globalParam := global.GlobalModel
	momentum := global.Momentum

	if chain.ValidCandidateBlock(n.Blockchain.LastBlock(), data, globalParam, momentum, n.Task.ModelSize, n.Task.CompModelSize,
		n.Task.PartNum, n.Task.Rank, n.Task.Beta, n.Task.Slr, n.Task.UnlabeledData, n.Task.Model) {
		if n.Malicious {
			rw.WriteString("N\n") //send No vote back
			rw.Flush()
			log.Printf("Node %d: Invalid candidate block.\n", n.ID)
			return
		}
		msg := chain.BlockToByteWithoutSig(data)
		sig, err := ecdsa.SignASN1(rand.Reader, n.Sig, msg)
		if err != nil {
			log.Printf("Node %d: Cannot sign the block. \n", n.ID)
		}
		rw.WriteString("Y\n") //send Yes vote back
		signature := chain.Signature{sig, n.Sig.PublicKey}

		//panic: gob: type not registered for interface: elliptic.p256Curve
		gob.Register(elliptic.P256())
		enc := gob.NewEncoder(rw)
		err = enc.Encode(signature)
		if err != nil {
			log.Printf("Node %d: Error encoding the signature: %s\n", n.ID, err)
		}
		rw.Flush()
		log.Printf("Node %d: Valid candidate block.\n", n.ID)
	} else {
		if n.Malicious {
			msg := chain.BlockToByteWithoutSig(data)
			sig, err := ecdsa.SignASN1(rand.Reader, n.Sig, msg)
			if err != nil {
				log.Printf("Node %d: Cannot sign the block. \n", n.ID)
			}
			rw.WriteString("Y\n") //send Yes vote back
			signature := chain.Signature{sig, n.Sig.PublicKey}

			//panic: gob: type not registered for interface: elliptic.p256Curve
			gob.Register(elliptic.P256())
			enc := gob.NewEncoder(rw)
			err = enc.Encode(signature)
			if err != nil {
				log.Printf("Node %d: Error encoding the signature: %s\n", n.ID, err)
			}
			rw.Flush()
			log.Printf("Node %d: Valid candidate block.\n", n.ID)
			return
		}
		rw.WriteString("N\n") //send No vote back
		rw.Flush()
		log.Printf("Node %d: Invalid candidate block.\n", n.ID)
	}
}

func (n *Node) handleChain(rw *bufio.ReadWriter, Raddr net.Addr) {
	log.Printf("Node %d: Receive chain info request from peer.\n", n.ID)
	enc := gob.NewEncoder(rw)
	err := enc.Encode(n.Blockchain.Chain)
	if err != nil {
		log.Printf("Node %d: Error encoding the local chain info.\n", n.ID)
	}
	rw.Flush()
}
