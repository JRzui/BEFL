package gopy

import "github.com/DataDog/go-python3"

//the imported module seems not able to be imported multiple times
//errors appear when import the module the third time
//not sure the reason
//centralized initialize the funcs instead
var (
	Interact        *python3.PyObject
	Init            *python3.PyObject
	GetModel_params *python3.PyObject
	Client_run      *python3.PyObject
	Node_agg        *python3.PyObject
	Test            *python3.PyObject
)
