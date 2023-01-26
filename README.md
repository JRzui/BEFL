## Environment
The golang environment, the latest go could be find at https://golang.org/

## Design description
The current prototype consists of four parts: chain, consensus, network and the node operation.
The chain package: 
This package defines the structure of the current blockchain, the linear chain part, custom block structure, custom transaction structure and the peer information.
The consensus package: 
The consensus to be implemented is the longest valid chain with block finalization by BFT voting.
The networking package:
This package enables the communication between nodes over tcp layer. A super node is introduced to mimic the broadcasting channel and the centralized control of the voting process.
The node package:
This package defines node’s operations. 


## Project running
First start the gossip network by 
```
go run main.go
```

## The go-python library
We use the package https://github.com/DataDog/go-python3. ```pkg-config``` is required to install the package.

Windows:

The ```pkg-config``` installation could be found https://github.com/DataDog/go-python3/issues/24

