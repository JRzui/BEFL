package main

import (
	"fmt"
	"log"
	"net"
	"net/http"
	"net/rpc"
	"os"
	"strconv"
	"time"

	"github.com/DataDog/go-python3"
	"github.com/JRzui/BCFedMI/chain"
	"github.com/JRzui/BCFedMI/client"
	"github.com/JRzui/BCFedMI/gopy"
	"github.com/JRzui/BCFedMI/network"
	"github.com/JRzui/BCFedMI/node"
	"github.com/JRzui/BCFedMI/run"
)

func main() {
	start_time := time.Now()
	file, _ := os.OpenFile("log/log.log", os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	log.SetOutput(file)

	bcNet := network.NetworkInit()
	rpc.Register(bcNet)
	rpc.HandleHTTP()
	l, _ := net.Listen("tcp", chain.BaseNode)
	go http.Serve(l, nil)

	//call
	conn, err := rpc.DialHTTP("tcp", chain.BaseNode)
	if err != nil {
		log.Fatal("dialing:", err)
	}

	//Nodes generation
	nodesNum := 100
	nodes := make([]*node.Node, 0)
	for i := 0; i < nodesNum; i++ {
		addr := "127.0.0.1:" + strconv.Itoa(30000+i)
		n := node.CreateNode(i, addr)
		nodes = append(nodes, n)
		//register to the blockchain network
		var reg bool
		conn.Call("BlockchainNetwork.Register", network.RegisterInfo{n.ID, n.Address, n.Vrf.RolesPk}, &reg)

		go n.ClientServing(n.Address)
	}

	//python initialize
	python3.Py_Initialize()

	gopy.Interact = gopy.ImportModule("fl", "interact")
	gopy.Init = gopy.GetFunc(gopy.Interact, "init")
	defer gopy.Init.DecRef()
	gopy.GetModel_params = gopy.GetFunc(gopy.Interact, "getModel_params")
	defer gopy.GetModel_params.DecRef()
	gopy.Client_run = gopy.GetFunc(gopy.Interact, "client_run")
	defer gopy.Client_run.DecRef()
	gopy.Node_agg = gopy.GetFunc(gopy.Interact, "node_agg")
	defer gopy.Node_agg.DecRef()
	gopy.Test = gopy.GetFunc(gopy.Interact, "test")
	defer gopy.Test.DecRef()

	//FL clients generation
	clients, test_data, unlabel, model, size, globalParam := client.CreateClients(client.ClientsNum)

	task := run.NewTask(model, unlabel, size, globalParam)
	run.TaskPublish(task, bcNet)

	//nodes get task info from network
	if bcNet.NewTask {
		run.NodesGetTask(nodes, bcNet)
	}

	state := python3.PyEval_SaveThread()

	for r := 0; r < client.Round; r++ {
		fmt.Println("----------------------------------------------------------------------------------")
		//committee constitution
		//nodes get roles in current round (whether become the members of the committee)
		for {
			run.ProcessCommittee(nodes, conn)
			if bcNet.CommitteeSetup {
				break
			}
		}

		run.NodesCommitteeUpdate(nodes, conn)

		//block prepare (candidate block generation)
		//FL clients local training
		run.ProcessFL(clients, r, nodes, conn)
		for {
			run.ProcessBlockPre(nodes, r, conn)
			if bcNet.BlockReceived {
				break
			}
		}

		//achieve consensus, vote
		for {
			run.ProcessBlock(bcNet, nodes, test_data, conn)
			if bcNet.NewBlock {
				break
			}
		}

		run.ProcessNextRound(nodes, conn)

	}

	run_time := time.Since(start_time)
	fmt.Println("Run time for 200 nodes: ", run_time)
	python3.PyEval_RestoreThread(state)
	python3.Py_Finalize()
}
