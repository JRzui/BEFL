package gopy

import "github.com/DataDog/go-python3"

//the imported module seems not able to be imported multiple times
//errors appear when import the module the third time
//not sure the reason
//centralized initialize the funcs instead
var (
	Client   *python3.PyObject // the python file
	Interact *python3.PyObject // the python file
	Init     *python3.PyObject
	//GetModel_params *python3.PyObject
	Honest_run   *python3.PyObject
	Attacker_run *python3.PyObject
	Node_run     *python3.PyObject
	Test         *python3.PyObject
	LF           *python3.PyObject // label flipping attacker
	BF           *python3.PyObject // bit flipping attacker
	Worker       *python3.PyObject //honest worker
)
