package node

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"log"
	"net/rpc"

	"github.com/JRzui/BEFL/chain"
	"github.com/JRzui/BEFL/consensus"
	"github.com/JRzui/BEFL/network"
)

type Node struct {
	ID              int
	Vrf             consensus.VRF
	Sig             *ecdsa.PrivateKey
	Address         string
	Blockchain      *chain.Blockchain
	Task            network.TaskInfo
	CommitteeMember bool
	Malicious       bool
}

func CreateNode(id int, addr string, malicious bool) *Node {
	vrf := consensus.VRF{}
	vrf.Init()
	sig, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	node := &Node{
		ID:              id,
		Vrf:             vrf,
		Sig:             sig,
		Address:         addr,
		Blockchain:      chain.NewBlockchain(), //since the large memory cost of maintaing the chain, we maintain a single blockchain in the simulation
		CommitteeMember: false,
		Malicious:       malicious,
	}
	return node
}

func (n *Node) ClientServing(port string) error {
	endpoint := network.NewEndpoint()

	//Add the handle funcs
	endpoint.AddHandleFunc("Vote", n.handleVote)
	endpoint.AddHandleFunc("Chain", n.handleChain)

	//Start listening
	return endpoint.Listen(port)
}

func (n *Node) SendTx(tx chain.LocalTransaction, conn *rpc.Client) {
	var sent bool
	err := conn.Call("BlockchainNetwork.SendTx", tx, &sent)
	if err != nil || sent != true {
		log.Printf("node %d unable to send tx to the network", n.ID)
	}
	log.Printf("Node %d sent pending tx to the network.\n", n.ID)
}

func (n *Node) GetTxs(conn *rpc.Client) {
	var txs []chain.LocalTransaction
	err := conn.Call("BlockchainNetwork.GetTxs", n.ID, &txs)
	if err != nil {
		log.Println("Cannot get pending transactions in the network.")
		return
	}

	for _, tx := range txs {
		//check if the transaction is the latest round model updates
		if tx.Round == int(n.Blockchain.LastBlock().Index)+1 {
			n.Blockchain.AddTransaction(tx)
			log.Printf("Node %d: Get pending txs from network\n", n.ID)
		}
	}
}

func (n *Node) GetGlobalParam(task string) (string, int) {
	var global string
	var round int

	if n.Blockchain.LastBlock().Index == 0 {
		global = n.Task.GlobalModel
		round = 0
	} else {
		global = n.Blockchain.LastBlock().Transactions[0].GlobalModel
		round = int(n.Blockchain.LastBlock().Transactions[0].Round)
	}

	return global, round
}

func (n *Node) AmMember() bool {
	_, found := n.Blockchain.Committee.Set[n.ID]
	return found
}
