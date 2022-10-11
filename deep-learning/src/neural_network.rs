use std::{fmt,ops};
use std::ops::Deref;
use std::cell::{Ref,RefCell};
use std::rc::Rc;
use std::collections::HashMap;
use linear_transform::tensor::tensor_base::Tensor;
use num;

#[derive(Debug, Clone)]
struct Neuron<T>
where T:num::Float + Clone {
    name: String,
    signal: Rc<RefCell<Tensor<T>>>,
    grad: Option<Tensor<T>>
}

impl<T> fmt::Display for Neuron<T>
where T: fmt::Display + Clone + num::Float {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
	let mut disp = format!("Neuron. name:{}\n",self.name);
	disp = format!("{}{}", disp, self.signal.borrow());
	write!(f,"{}",disp)
    }
}

impl<T> Neuron<T>
where T:num::Float + Clone {

    pub fn create(name:&str, init_signal:Tensor<T>) -> Neuron<T> {
	Neuron {
	    name: name.to_string(),
	    signal: Rc::new(RefCell::new(init_signal)),
	    grad: None,
	}
    }

    pub fn get_signal(&self) -> Rc<RefCell<Tensor<T>>> {
	Rc::clone(&self.signal)
    }

    pub fn name(&self) -> &str {
	&self.name
    }

    pub fn assign(&mut self, signal:Tensor<T>) -> () {
	(*self.signal.borrow_mut()) = signal;
    }

}

struct Synapse<T>
    where T:num::Float + Clone {
    forward : fn (inputs: &Vec<&Tensor<T>>) -> Vec<Tensor<T>>,
    backward: fn (inputs: &Vec<&Tensor<T>>) -> Vec<Tensor<T>>,
}

impl<T> Synapse<T>
where T:num::Float + Clone {

    pub fn new(forward : fn (inputs: &Vec<&Tensor<T>>) -> Vec<Tensor<T>>,
	       backward: fn (inputs: &Vec<&Tensor<T>>) -> Vec<Tensor<T>>) -> Synapse<T> {
	Synapse {
	    forward,
	    backward,
	}
    }

    pub fn forward(&mut self,inputs: &Vec<Rc<RefCell<Tensor<T>>>>) -> Vec<Tensor<T>> {
	let holder:Vec<Ref<'_,Tensor<T>>> = inputs.iter().map(|i| { i.borrow() }).collect();
	let inputs:Vec<&Tensor<T>> = holder.iter().map(|h| { h.deref() }).collect();
	((*self).forward)(&inputs)
    }

    pub fn backward(&mut self, inputs: &Vec<&Tensor<T>>) -> Vec<Tensor<T>> {
	((*self).backward)(inputs)
    }

}

struct SynapseNode<T>
where T:num::Float + Clone {
    name: String,
    generation: usize,
    inputs: Vec<Rc<RefCell<Neuron<T>>>>,
    outputs: Vec<Rc<RefCell<Neuron<T>>>>,
    synapse: Synapse<T>
}

impl<T> SynapseNode<T>
where T:num::Float + Clone {

    pub fn new(prev:Option<Rc<RefCell<SynapseNode<T>>>>,
	       name: &str,
	       inputs: Vec<Rc<RefCell<Neuron<T>>>>,
	       outputs: Vec<Rc<RefCell<Neuron<T>>>>,
	       forward : fn (inputs: &Vec<&Tensor<T>>) -> Vec<Tensor<T>>,
	       backward: fn (inputs: &Vec<&Tensor<T>>) -> Vec<Tensor<T>>) -> SynapseNode<T> {

	let s = Synapse::<T>::new(forward, backward);
	let mut generation:usize = 0;

	if let Some(p) = prev {
	    generation = p.borrow().generation() + 1;
	}

	SynapseNode {
	    name: name.to_string(),
	    generation,
	    inputs,
	    outputs,
	    synapse :s
	}
    }

    pub fn generation(&self) -> usize {
	self.generation
    }

    pub fn forward_prop(&mut self) -> () {
	let inputs:Vec<Rc<RefCell<Tensor<T>>>> = self.inputs.iter().map(|n| n.borrow().get_signal()).collect();
	let outputs = self.synapse.forward(&inputs);
	for (os,on) in self.outputs.iter().zip(outputs.iter()) {
	    os.borrow_mut().assign(on.clone());
	}
    }
    
}

struct NeuralNetwork<T>
where T:num::Float + Clone {
    neurons:  HashMap<String, Rc<RefCell<Neuron<T>>>>,
    synapse_nodes: HashMap<String, Rc<RefCell<SynapseNode<T>>>>,
    input_neurons: Vec<Rc<RefCell<Neuron<T>>>>,
    generation_table: Vec<Vec<Rc<RefCell<SynapseNode<T>>>>>
}

impl<T> NeuralNetwork<T>
where T:num::Float + Clone {

    pub fn new() -> NeuralNetwork<T> {
	NeuralNetwork {
	    neurons: HashMap::new(),
	    synapse_nodes: HashMap::new(),
	    input_neurons: Vec::new(),
	    generation_table: Vec::new()
	}
    }

    pub fn create_neuron(&mut self, name:&str, input: bool) {
	let n = Rc::new(RefCell::new(Neuron::<T>::create(name, Tensor::<T>::zero(&[1,1]))));
	self.neurons.insert(name.to_string(), Rc::clone(&n));
	if input {
	    self.input_neurons.push(Rc::clone(&n))
	}
    }

    pub fn add_neuron(&mut self, n:Neuron<T>) -> Rc<RefCell<Neuron<T>>> {
	let name = n.name().to_string();
	let rcn = Rc::new(RefCell::new(n));
	self.neurons.insert(name, Rc::clone(&rcn));
	rcn
    }

    pub fn create_synapse(&mut self,
		       prev : Option<&str>,
		       name: &str,
		       inputs: Vec<&str>,
		       outputs: Vec<&str>,
		       forward : fn (inputs: &Vec<&Tensor<T>>) -> Vec<Tensor<T>>,
		       backward: fn (inputs: &Vec<&Tensor<T>>) -> Vec<Tensor<T>>) {
	let input_neurons:Vec<Rc<RefCell<Neuron<T>>>> =
	    inputs.iter().map(|label| Rc::clone(&self.neurons[*label])).collect();
	let output_neurons:Vec<Rc<RefCell<Neuron<T>>>> =
	    outputs.iter().map(|label| Rc::clone(&self.neurons[*label])).collect();

	let sn: SynapseNode<T> = if let Some(p) = prev {
	    let prev_node = Rc::clone(&self.synapse_nodes[p]);
	    SynapseNode::new(Some(prev_node), name, input_neurons, output_neurons, forward, backward)
	}
	else {
	    SynapseNode::new(None, name, input_neurons, output_neurons, forward, backward)
	};

	let rsn = Rc::new(RefCell::new(sn));
	let g = rsn.borrow().generation();

	if self.generation_table.len() < g+1 {
	    self.generation_table.resize(g+1, Vec::new());
	}
	self.generation_table[g].push(Rc::clone(&rsn));
	self.synapse_nodes.insert(name.to_string(), rsn);
    }

    pub fn ref_neuron(&self, name:&str) -> &Rc<RefCell<Neuron<T>>> {
	&self.neurons[name]
    }

    pub fn forward_prop(&mut self, inputs: Vec<Tensor<T>>) -> Result<Vec<Tensor<T>>,String> {

	for (i,n) in inputs.iter().zip(self.input_neurons.iter()) {
	    n.borrow_mut().assign(i.clone())
	}

	for g in self.generation_table.iter() {
	    for s in g.iter() {
		s.borrow_mut().forward_prop()
	    }
	}

	Ok(vec!())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nn_test_add() {
	let mut nn = NeuralNetwork::<f32>::new();
	let n1 = Tensor::<f32>::from_array(&[1,1], &[1.0]);
	let n2 = Tensor::<f32>::from_array(&[1,1], &[1.0]);

	nn.create_neuron("n1",true);
	nn.create_neuron("n2",true);
	nn.create_neuron("n3",false);

	nn.create_synapse(None,
			  "add1", vec!["n1", "n2"], vec!["n3"],
			  | inputs | -> Vec<Tensor<f32>> {
			      vec![inputs[0] + inputs[1]]
			  },
			  | inputs | -> Vec<Tensor<f32>> {
			      vec!()
			  });

	nn.forward_prop(vec![n1,n2]);

	println!("n3 {:?}", nn.ref_neuron("n3"));
    }
}

/*
#[derive(Debug, Clone)]
struct Neuron<T>
where T:num::Float + Clone {
    name: String,
    signal: Tensor<T>,
    grad: Option<Tensor<T>>
}

#[derive(Debug, Clone)]
struct Synapse<'a, T>
where T:num::Float + Clone {
    name: String,
    inputs:  Vec<&'a Tensor<T>>,
    outputs: Vec<&'a Tensor<T>>
}

enum NNLeaf<'a, T>
where T: num::Float {
    Synapse(Synapse<'a,T>),
    Neuron(Neuron<T>)
}

struct NNNode<'a,

impl<T> fmt::Display for Neuron<T>
where T: fmt::Display + Clone + num::Float {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
	let mut disp = format!("Neuron. name:{}\n",self.name);
	disp = format!("{}{}", disp, self.signal);
	write!(f,"{}",disp)
    }
}

impl<T> Neuron<T>
where T:num::Float + Clone {

    pub fn create(new_name:String, init_signal:Tensor<T>) -> Neuron<T> {
	Neuron {
	    name: new_name,
	    signal: init_signal,
	    grad: None,
	}
    }
}

impl<'a, T> Synapse<'a,T>
where T:num::Float + Clone {
    pub fn create(new_name: String) -> Synapse<'a,T> {
	Synapse {
	    name: new_name,
	    inputs: Vec::new(),
	    outputs: Vec::new()
	}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nn_test(){
	let n1 = Neuron::<f32>::create("n1".to_string(), Tensor::<f32>::from_array(&[1,1],&[1.0]));
	let n2 = Neuron::<f32>::create("n2".to_string(), Tensor::<f32>::from_array(&[1,1],&[1.0]));
	println!("{},{}",n1,n2);
    }
}
*/
