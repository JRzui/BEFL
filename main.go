package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/rpc"
	"os"
	"strconv"
	"time"

	"github.com/DataDog/go-python3"
	"github.com/JRzui/BEFL/chain"
	"github.com/JRzui/BEFL/client"
	"github.com/JRzui/BEFL/gopy"
	"github.com/JRzui/BEFL/network"
	"github.com/JRzui/BEFL/node"
	"github.com/JRzui/BEFL/run"
)

func main() {
	path := "log"
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		err := os.Mkdir(path, os.ModePerm)
		if err != nil {
			log.Println(err)
		}
	}
	path := "results"
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		err := os.Mkdir(path, os.ModePerm)
		if err != nil {
			log.Println(err)
		}
	}

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
	nodes := make([]*node.Node, 0)
	Sybils := int(float64(chain.NodeNum) * chain.SybilRatio)
	//Honest blockchain nodes
	for i := 0; i < chain.NodeNum-Sybils; i++ {
		addr := "127.0.0.1:" + strconv.Itoa(40000+i)
		n := node.CreateNode(i, addr, false)
		nodes = append(nodes, n)
		//register to the blockchain network
		var reg bool
		conn.Call("BlockchainNetwork.Register", network.RegisterInfo{n.ID, n.Address, n.Vrf.RolesPk}, &reg)

		go n.ClientServing(n.Address)
	}
	//Malicious blockchain nodes
	for i := chain.NodeNum - Sybils; i < chain.NodeNum; i++ {
		addr := "127.0.0.1:" + strconv.Itoa(40000+i)
		n := node.CreateNode(i, addr, true)
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
	gopy.Honest_run = gopy.GetFunc(gopy.Interact, "honest_run")
	defer gopy.Honest_run.DecRef()
	gopy.Attacker_run = gopy.GetFunc(gopy.Interact, "attacker_run")
	defer gopy.Attacker_run.DecRef()
	gopy.Node_run = gopy.GetFunc(gopy.Interact, "node_run")
	defer gopy.Node_run.DecRef()
	gopy.Malicious_node_run = gopy.GetFunc(gopy.Interact, "malicious_node_run")
	defer gopy.Malicious_node_run.DecRef()
	gopy.Test = gopy.GetFunc(gopy.Interact, "test")
	defer gopy.Test.DecRef()

	gopy.Client = gopy.ImportModule("fl", "client")
	gopy.LF = gopy.GetFunc(gopy.Client, "LF")
	defer gopy.LF.DecRef()
	gopy.BF = gopy.GetFunc(gopy.Client, "BF")
	defer gopy.BF.DecRef()
	gopy.Worker = gopy.GetFunc(gopy.Client, "Worker")
	defer gopy.Worker.DecRef()

	//FL clients generation
	attackers, workers, test_data, unlabel, model, size, comp_size, globalParam, momentum := client.CreateClients(client.Attack)
	task := run.NewTask(model, unlabel, size, comp_size, globalParam, momentum, client.Rank, client.Beta, client.Slr)
	run.TaskPublish(task, bcNet)

	//nodes get task info from network
	if bcNet.NewTask {
		run.NodesGetTask(nodes, bcNet)
	}

	//bootstrap
	go func() {
		bcNet.CommitteeWait <- true
		bcNet.NewRound <- true
		bcNet.CandidateWait <- true
	}()

	fmt.Printf("---------------------------Round %d----------------------------\n", task.CurrentRound)
	var state *python3.PyThreadState
	var vote_round int
	var committee_time time.Duration
	var vote_time time.Duration
	var cand_gen_time time.Duration
	var propagation_time time.Duration
	Start := time.Now()
	for task.CurrentRound < client.Round {
		select {
		case <-bcNet.CommitteeWait:
			start := time.Now()
			run.ProcessCommittee(nodes, conn, bcNet)
			stop := time.Since(start)
			committee_time += stop
		case <-bcNet.CommitteeSetup:
			run.NodesCommitteeUpdate(nodes, conn, bcNet)
		case <-bcNet.NewRound:
			vote_round = 0
			state = python3.PyEval_SaveThread()
			run.ProcessFL(workers, attackers, task.CurrentRound, nodes, conn)
		case <-bcNet.CandidateWait:
			start := time.Now()
			run.ProcessBlockPre(nodes, task.CurrentRound, conn, bcNet)
			stop := time.Since(start)
			cand_gen_time += stop
		case <-bcNet.BlockReceived:
			start := time.Now()
			run.ProcessBlock(bcNet, nodes, test_data, conn)
			stop := time.Since(start)
			vote_time += stop
			vote_round++
			if vote_round >= chain.MaxVoteStep {
				//recounstruct the committee
				bcNet.Members = chain.NewMemberSet() // clear current committee, waiting for the next round committee constitution
				go func() { bcNet.CommitteeWait <- true }()
				vote_round = 0
			}
		case <-bcNet.NewBlock:
			start := time.Now()
			run.ProcessNextRound(bcNet, nodes, conn, &task.CurrentRound)
			stop := time.Since(start)
			propagation_time += stop
			python3.PyEval_RestoreThread(state)
			run.SaveStakeMap(nodes, bcNet.StakeMap)
			fmt.Printf("---------------------------Round %d-----------------------------\n", task.CurrentRound)
		case <-time.After(chain.BlockTimeout):
			fmt.Println("Block generation time out")
		}
	}
	round_time := time.Since(Start)
	
	f, err := os.OpenFile("results/time.csv", os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		panic(err)
	}
	writer := csv.NewWriter(f)
	writer.Write([]string{fmt.Sprintf("%v", committee_time)})
	writer.Write([]string{fmt.Sprintf("%v", cand_gen_time)})
	writer.Write([]string{fmt.Sprintf("%v", vote_time)})
	writer.Write([]string{fmt.Sprintf("%v", propagation_time)})
	writer.Write([]string{fmt.Sprintf("%v", round_time)})
	writer.Flush()
	f.Close()

	python3.Py_Finalize()
}
