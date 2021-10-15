package client

var (
	Task       = "MNIST"
	P          = 0.2  //the non-i.i.d extend of MNIST dataset
	K          = 5    //local training step
	B          = 64   //batch size
	Lr         = 0.25 //learning rate
	Round      = 10
	C          = 0.2 //the fraction of participated clients
	Cm         = 0.1 //the fraction of malicious clients existed per round
	ClientsNum = 100
	PartNum    = int(float64(ClientsNum) * C) //the number of participants in each round
)
