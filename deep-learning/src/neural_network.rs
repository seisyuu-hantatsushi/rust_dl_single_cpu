use std::fmt;
use std::cell::{Ref,RefCell};
use std::rc::Rc;
use linear_transform::tensor::tensor_base::Tensor;

use openssl::sha::sha512;

struct Neuron<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone {
    hashed_id: [u8;64],
    name: String,
    signal: Tensor<T>,
    generator: Option<NNSynapseNode<T>>
}

impl<T> fmt::Display for Neuron<T>
where T: fmt::Display + Clone + num::Float + num::pow::Pow<T, Output = T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
	let mut disp = format!("Neuron. name:{}\n",self.name);
	disp = format!("{}Singal:{}", disp, self.signal);
	write!(f,"{}",disp)
    }
}

impl<T> Neuron<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone {
    pub fn new(name:&str, init_signal:Tensor<T>) -> Neuron<T> {
	Neuron {
	    hashed_id: sha512(name.as_bytes()),
	    name: name.to_string(),
	    signal: init_signal,
	    generator: None
	}
    }

    pub fn name(&self) -> &str {
	&self.name
    }

    pub fn set_generator(&mut self, s:NNSynapseNode<T>){
	self.generator = Some(Rc::clone(&s))
    }

    pub fn ref_generator(&self) -> &Option<NNSynapseNode<T>> {
	&self.generator
    }

    pub fn assign(&mut self, signal:Tensor<T>) -> () {
	self.signal = signal;
    }

    pub fn ref_signal(&self) -> &Tensor<T> {
	&self.signal
    }
}

type NNNeuron<T> = Rc<RefCell<Neuron<T>>>;

fn nn_neuron_new<T>(name:&str, init_signal:Tensor<T>) -> NNNeuron<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone {
    Rc::new(RefCell::new(Neuron::<T>::new(name,init_signal)))
}

type MakeDiffNode<T> = fn (inputs: &Vec<NNNeuron<T>>, grads: Vec<NNNeuron<T>>) -> (NNSynapseNode<T>, Vec<NNNeuron<T>>);

struct Synapse<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone {
    forward: fn (inputs: Vec<&Tensor<T>>) -> Vec<Tensor<T>>,
    make_diff_node: MakeDiffNode<T>
}

impl<T> Synapse<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone {
    fn new(forward: fn (inputs: Vec<&Tensor<T>>) -> Vec<Tensor<T>>,
	   make_diff_node: MakeDiffNode<T> ) -> Synapse<T> {
	Synapse {
	    forward,
	    make_diff_node
	}
    }
}

struct SynapseNode<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone {
    hashed_id: [u8;64],
    name: String,
    inputs: Vec<NNNeuron<T>>,
    outputs: Vec<NNNeuron<T>>,
    synapse: Synapse<T>,
    generation: usize
}

type NNSynapseNode<T> = Rc<RefCell<SynapseNode<T>>>;

impl<T> SynapseNode<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone {

    pub fn new(name:&str,
	       inputs: Vec<NNNeuron<T>>,
	       outputs: Vec<NNNeuron<T>>,
	       synapse: Synapse<T>) -> SynapseNode<T> {
	let max_input_generation:usize =
	    inputs.iter().fold(0, |g,ni| {
		let generation = 
		    if let Some(generator) = ni.borrow().ref_generator() {
			generator.borrow().get_generation()
		    }
		else {
		   0
		};
		generation
	    });
	SynapseNode {
	    hashed_id: sha512(name.as_bytes()),
	    name: name.to_string(),
	    inputs,
	    outputs,
	    synapse,
	    generation: max_input_generation+1
	}
    }

    pub fn get_generation(&self) -> usize {
	self.generation
    }

    pub fn set_generation(&mut self, generation:usize) {
	self.generation = generation
    }

    fn twoway_branch(input: NNNeuron<T>, outputs: Vec<NNNeuron<T>>) -> (NNSynapseNode<T>,Vec<NNNeuron<T>>) {
	let label = input.borrow().name().to_string() + " branch";
	let outputs_clone = outputs.iter().map(|o| Rc::clone(o)).collect();
	let s = SynapseNode::<T>::new(&label,
				      vec![Rc::clone(&input)],
				      outputs_clone,
				      Synapse {
					  forward: |inputs| {
					      vec![inputs[0].clone(),inputs[0].clone()]
					  },
					  make_diff_node: |_inputs, grads| {
					      Self::add(Rc::clone(&grads[0]), Rc::clone(&grads[0]))
					  }
				      });
	let rs = Rc::new(RefCell::new(s));
	for no in outputs.iter() {
	    no.borrow_mut().set_generator(Rc::clone(&rs));
	}
	(rs, outputs)
    }

    fn add(x:NNNeuron<T>, y:NNNeuron<T>) -> (NNSynapseNode<T>,Vec<NNNeuron<T>>) {
    	let label = "(".to_string()+x.borrow().name() + "+" + y.borrow().name() +")";
	let output = Rc::new(RefCell::new(Neuron::<T>::new(&label, Tensor::<T>::zero(&[1,1]))));
	let s = SynapseNode::<T>::new(&label,
				      vec![Rc::clone(&x),Rc::clone(&y)],
				      vec![Rc::clone(&output)],
				      Synapse {
					  forward: |inputs| {
					      vec![inputs[0] + inputs[1]]
					  },
					  make_diff_node: |inputs, grads| {
					      let outputs:Vec<NNNeuron<T>> =
						  inputs.iter().map(|ni| {
						      let label = "g/".to_string() + ni.borrow().name();
						      Rc::new(RefCell::new(Neuron::<T>::new(&label, Tensor::<T>::zero(&[1,1]))))
						  }).collect();
					      Self::twoway_branch(Rc::clone(&grads[0]),outputs)
					  }
				      });
	let rs = Rc::new(RefCell::new(s));
	output.borrow_mut().set_generator(Rc::clone(&rs));
	(rs, vec![output])
    }

    fn forward(&self) -> Vec<NNNeuron<T>> {
	let inputs_holder = self.inputs.iter().map(|n| n.borrow()).collect::<Vec<Ref<'_,Neuron<T>>>>();
	let inputs = inputs_holder.iter().map(|n| n.ref_signal()).collect::<Vec<&Tensor<T>>>();
	let outputs = (self.synapse.forward)(inputs);
	self.outputs.iter().zip(outputs.into_iter()).map(|(n,t)| {n.borrow_mut().assign(t); Rc::clone(n)}).collect()
    }

    fn make_diff_node(&self, grads:Vec<NNNeuron<T>>) -> (NNSynapseNode<T>,Vec<NNNeuron<T>>) {
	(self.synapse.make_diff_node)(&self.inputs, grads)
    }

}

struct NeuralNetwork<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone {
    generation_table: Vec<NNSynapseNode<T>>
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nn_test_add() {
	{
	    let x = nn_neuron_new::<f64>("x", Tensor::<f64>::from_array(&[1,1],&[1.0]));
	    let z1 = nn_neuron_new::<f64>("z1", Tensor::<f64>::from_array(&[1,1],&[0.0]));
	    let z2 = nn_neuron_new::<f64>("z2", Tensor::<f64>::from_array(&[1,1],&[0.0]));

	    let (twoway,_outputs) = SynapseNode::<f64>::twoway_branch(x,vec![z1,z2]);
	    let zs = twoway.borrow().forward();
	    println!("zs[0] = {}", zs[0].borrow());
	    println!("zs[1] = {}", zs[1].borrow());
	    let gz1 = nn_neuron_new::<f64>("gz1", Tensor::<f64>::from_array(&[1,1],&[1.0]));
	    let gz2 = nn_neuron_new::<f64>("gz2", Tensor::<f64>::from_array(&[1,1],&[1.0]));

	    let (backprop, _outputs) = twoway.borrow().make_diff_node(vec![gz1,gz2]);
	    let gs = backprop.borrow().forward();
	    println!("gs[0] = {}", gs[0].borrow());
	}
	{
	    let x = nn_neuron_new::<f64>("x", Tensor::<f64>::from_array(&[1,1],&[1.0]));
	    let y = nn_neuron_new::<f64>("y", Tensor::<f64>::from_array(&[1,1],&[1.0]));
	    let (add,_outputs) = SynapseNode::<f64>::add(x,y);
	    let z = add.borrow().forward();
	    println!("z[0] = {}", z[0].borrow());
	    let gz = nn_neuron_new::<f64>("gz", Tensor::<f64>::from_array(&[1,1],&[1.0]));
	    let (backprop, _outputs) = add.borrow().make_diff_node(vec![gz]);
	    let gs = backprop.borrow().forward();
	    println!("gs[0] = {}", gs[0].borrow());
	    println!("gs[1] = {}", gs[1].borrow());
	    let (backprop, _outputs) = backprop.borrow().make_diff_node(gs);
	    let gs = backprop.borrow().forward();
	    println!("gs[0] = {}", gs[0].borrow());
	}
	{
	    let x = nn_neuron_new::<f64>("x", Tensor::<f64>::from_array(&[1,1],&[1.0]));
	    let y = nn_neuron_new::<f64>("y", Tensor::<f64>::from_array(&[1,1],&[1.0]));
	    let z = nn_neuron_new::<f64>("z", Tensor::<f64>::from_array(&[1,1],&[1.0]));
	    let (node1,outputs) = SynapseNode::<f64>::add(x,y);
	    let xy = Rc::clone(&outputs[0]);
	    let (node2,outputs) = SynapseNode::<f64>::add(z,xy);
	    node1.borrow().forward();
	    node2.borrow().forward();
	    println!("outputs[0] = {}", outputs[0].borrow());
	}
    }
}
