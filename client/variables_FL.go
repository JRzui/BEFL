package client

var (
	Task       = "cifar"
	Attack     = "None" //laber flipping attack
	K          = 5      //local training step
	B          = 64     //batch size
	Beta       = 0.9
	Lr         = 0.1 //client learning rate
	Slr        = 1.0 //server learning rate
	Round      = 50
	M          = 20  //the number of participants in each round
	Cm         = 0.0 //the fraction of malicious clients existed per round
	Rank       = 4   //the rank of the compressed matrix
	ClientsNum = 50
)
