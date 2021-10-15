package gopy

import "github.com/DataDog/go-python3"

func ImportModule(pyFolder string, pyFile string) *python3.PyObject {
	python3.PyRun_SimpleString("import sys\nsys.path.append(\"\")")

	Imodule := python3.PyImport_ImportModule(pyFolder + "." + pyFile) //new ref, decref needed
	defer Imodule.DecRef()
	module := python3.PyImport_AddModule(pyFolder + "." + pyFile) //borrowed ref from Imodue, do not need decref
	return module
}

func GetFunc(module *python3.PyObject, funcName string) *python3.PyObject {
	dict := python3.PyModule_GetDict(module)
	function := python3.PyDict_GetItemString(dict, funcName)
	if function == nil || (python3.PyCallable_Check(function) == false) {
		panic("Error: Not a callable function.")
	}
	return function
}

func TypeCheck(val *python3.PyObject) string {
	if python3.PyBool_Check(val) {
		return "bool"
	}
	if python3.PyByteArray_Check(val) {
		return "ByteArray"
	}
	if python3.PyBytes_Check(val) {
		return "Bytes"
	}
	if python3.PyComplex_Check(val) {
		return "Complex"
	}
	if python3.PyFloat_Check(val) {
		return "Float"
	}
	if python3.PyLong_Check(val) {
		return "Long"
	}
	if python3.PyTuple_Check(val) {
		return "Tuple"
	}
	if python3.PyList_Check(val) {
		return "List"
	}
	if python3.PyDict_Check(val) {
		return "Dict"
	}
	if python3.PyUnicode_Check(val) {
		return "Unicode"
	}
	if python3.PyModule_Check(val) {
		return "Module"
	}
	return "Not a valid type"
}

func ArgFromInt(arg int) *python3.PyObject {
	return python3.PyLong_FromGoInt(arg)
}

func ArgFromString(arg string) *python3.PyObject {
	return python3.PyUnicode_FromString(arg)
}

func ArgFromFloat(arg float64) *python3.PyObject {
	return python3.PyFloat_FromDouble(arg)
}

/*{list of list} --> {list of list}*/
func ArgFromListArray_Float(arg [][]float64) *python3.PyObject {
	newArg := python3.PyList_New(len(arg))
	for i := 0; i < len(arg); i++ {
		python3.PyList_SetItem(newArg, i, ArgFromList_Float(arg[i]))
	}
	return newArg
}

/*{list of list} --> {list of tuple}*/
func ArgFromListArray_Int(arg [][]int) *python3.PyObject {
	newArg := python3.PyList_New(len(arg))
	for i := 0; i < len(arg); i++ {
		python3.PyList_SetItem(newArg, i, ArgFromList_Int(arg[i]))
	}
	return newArg
}

func ArgFromList_Int(arg []int) *python3.PyObject {
	newArg := python3.PyTuple_New(len(arg))
	for i := 0; i < len(arg); i++ {
		python3.PyTuple_SetItem(newArg, i, ArgFromInt(arg[i]))
	}
	return newArg
}

func ArgFromList_Float(arg []float64) *python3.PyObject {
	newArg := python3.PyList_New(len(arg))
	for i := 0; i < len(arg); i++ {
		python3.PyList_SetItem(newArg, i, ArgFromFloat(arg[i]))
	}
	return newArg
}

func PyToBool(val *python3.PyObject) bool {
	if val == python3.Py_False {
		return false
	} else {
		return true
	}
}

func PyToFloat(val *python3.PyObject) float64 {
	return python3.PyFloat_AsDouble(val)
}

func PyToInt(val *python3.PyObject) int {
	return python3.PyLong_AsLong(val)
}

func PyListToFloat(val *python3.PyObject) []float64 {
	//create iterator
	iter := val.GetIter() //new ref
	defer iter.DecRef()

	next := iter.GetAttrString("__next__") //new ref, returns the next item or nil
	defer next.DecRef()

	var newArray []float64
	for i := 0; i < val.Length(); i++ {
		item := next.CallObject(nil)
		newArray = append(newArray, PyToFloat(item))
	}
	return newArray
}

func PyListList_Float(val *python3.PyObject) [][]float64 {
	var newArray [][]float64
	for i := 0; i < val.Length(); i++ {
		intreList := python3.PyList_GetItem(val, i)
		newArray = append(newArray, PyListToFloat(intreList))
	}
	return newArray
}

func PyTupleToInt(val *python3.PyObject) []int {
	vals := make([]int, 0)
	for i := 0; i < val.Length(); i++ {
		vals = append(vals, PyToInt(python3.PyTuple_GetItem(val, i)))
	}
	return vals
}

func PyListTuple_Int(val *python3.PyObject) [][]int {
	var Array [][]int
	for i := 0; i < val.Length(); i++ {
		tuple := python3.PyList_GetItem(val, i)
		Array = append(Array, PyTupleToInt(tuple))
	}
	return Array
}
