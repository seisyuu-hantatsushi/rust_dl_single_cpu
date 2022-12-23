/* -*- tab-width:4 -*- */
use std::{fmt,fs};
use std::io::Write;
use std::cell::RefCell;
use std::rc::Rc;
use std::collections::{BTreeSet,HashMap};
use linear_transform::tensor::Tensor;

use crate::neuron::{NeuronPrimType,NNNeuron,Neuron,nn_neuron_new,nn_neuron_constant};
use crate::synapse::{NNSynapseNode,SynapseNode};

struct ComputationalGraph<T>
where T: NeuronPrimType<T> {
	neurons: HashMap<* const RefCell<Neuron<T>>, NNNeuron<T>>,
	synapse_nodes: HashMap<* const RefCell<SynapseNode<T>>, Rc<RefCell<SynapseNode<T>>>>,
	generation_table: Vec<Vec<NNSynapseNode<T>>>
}

impl<T> ComputationalGraph<T>
where T:NeuronPrimType<T> {

	fn new() -> ComputationalGraph<T> {
		ComputationalGraph {
			neurons: HashMap::new(),
			synapse_nodes: HashMap::new(),
			generation_table: Vec::new()
		}
	}

	fn append_nodes(&mut self, ns:Vec<NNSynapseNode<T>>) {
		for node in ns.iter() {
			self.synapse_nodes.insert(Rc::as_ptr(node), Rc::clone(node));
			let g = node.borrow().get_generation();
			if self.generation_table.len() < g + 1 {
				self.generation_table.resize(g+1, Vec::new());
			}
			self.generation_table[g].push(Rc::clone(&node));
		}
	}

	fn append_neurons(&mut self, ns:Vec<NNNeuron<T>>) {
		for neuron in ns.iter() {
			self.neurons.insert(Rc::as_ptr(neuron), Rc::clone(neuron));
		}
	}

	fn erase_isolation_neuron(&mut self) {
		let mut linked_neurons:BTreeSet<* const RefCell<Neuron<T>>> = BTreeSet::new();
		let mut erase_keys:Vec<* const RefCell<Neuron<T>>> = vec!();
		for sn in self.synapse_nodes.values() {
			for ni in sn.borrow().inputs().iter() {
				linked_neurons.insert(Rc::as_ptr(ni));
			}
			for no in sn.borrow().outputs().iter() {
				linked_neurons.insert(Rc::as_ptr(no));
			}
		}
		for nk in self.neurons.keys() {
			if !linked_neurons.contains(nk) {
				erase_keys.push(*nk);
			}
		}
		for k in erase_keys.iter() {
			let _ = self.neurons.remove(k);
		}
	}

	fn forward(&mut self) -> Vec<NNNeuron<T>> {
		for g in self.generation_table.iter() {
			for node in g.iter() {
				let outputs = node.borrow().forward();
				/*
				for output in outputs.iter() {
					println!("{:p} {}", Rc::as_ptr(&output), output.borrow())
				}*/
			}
		}
		if let Some(ref gs) = self.generation_table.last() {
			let mut outputs:Vec<NNNeuron<T>> = vec!();
			for g in gs.iter() {
				for output in g.borrow().outputs().iter() {
					outputs.push(Rc::clone(&output));
				}
			}
			outputs
		}
		else {
			vec!()
		}
	}

	fn backward(&mut self) -> ComputationalGraph<T> {
		let mut cg = ComputationalGraph::<T>::new();
		for rg in self.generation_table.iter().rev() {
			for node in rg.iter() {
				let (sns,outputs) = node.borrow().make_diff_node();
				cg.append_nodes(sns);
				cg.append_neurons(outputs);
			}
		}
		cg.erase_isolation_neuron();
		cg
	}

	fn clear_grads(&mut self) {
		for nn in self.neurons.values() {
			nn.borrow_mut().clear_grad()
		}
	}

	pub fn make_dot_graph(&mut self, file_name:&str, disp_generation:bool) -> () {
		let mut output = "digraph g {\n".to_string();
		for n in self.neurons.values() {
			let nref = n.borrow();
			let id_str = format!("{:p}", Rc::as_ptr(&n));
			let color = if nref.is_constant() {
				"seagreen4"
			}
			else {
				"orange"
			};
			let label = nref.name().to_string()+"\n"+"id="+&id_str;
			output = output.to_string() + "\"" + &id_str + "\"" + " [label=\"" + &label + "\", color=" + &color + ", style=filled]" + "\n";
		}
		for sn in self.synapse_nodes.values() {
			let sn_ref = sn.borrow();
			let label = if disp_generation {
				sn_ref.name().to_string() + "\ngen=" + &sn_ref.get_generation().to_string()
			}
			else {
				sn_ref.name().to_string()
			};
			let id_str = format!("{:p}", Rc::as_ptr(&sn));
			output = output.to_string() + "\"" + &id_str + "\"" + " [label=\"" + &label + "\", color=lightblue, style=filled, shape=box ]" + "\n";
			for ni in sn_ref.inputs().iter() {
				let niid_str = format!("{:p}",Rc::as_ptr(&ni));
				output = output.to_string() + "\"" + &niid_str.to_string() + "\"->\"" + &id_str + "\"\n"
			}

			for no in sn_ref.outputs().iter() {
				let noid_str = format!("{:p}", Rc::as_ptr(&no));
				output = output.to_string() + "\"" + &id_str + "\"->\"" + &noid_str + "\"\n"
			}
		}

		output = output + "}";
		let mut ofs = fs::File::create(file_name).unwrap();
		if let Err(e) = ofs.write_all(output.as_bytes()){
			println!("{}",e)
		}
	}

}

pub struct NeuralNetwork<T>
where T: NeuronPrimType<T>{
	cg_order: Vec<ComputationalGraph<T>>
}

impl<T> NeuralNetwork<T>
where T:NeuronPrimType<T> {
	pub fn new() -> NeuralNetwork<T> {
		NeuralNetwork {
			cg_order: vec![ComputationalGraph::<T>::new()]
		}
	}

	pub fn create_neuron(&mut self, label:&str, init_signal: Tensor<T>) -> NNNeuron<T> {
		let nn = nn_neuron_new(label, init_signal);
		self.cg_order[0].append_neurons(vec![Rc::clone(&nn)]);
		nn
	}

	pub fn create_constant(&mut self, label:&str, init_signal: Tensor<T>) -> NNNeuron<T> {
		let nn = nn_neuron_constant(label, init_signal);
		self.cg_order[0].append_neurons(vec![Rc::clone(&nn)]);
		nn
	}

	pub fn reshape(&mut self, x:NNNeuron<T>, shape:Vec<usize>) -> NNNeuron<T> {
		let (sn,output) = SynapseNode::<T>::reshape(x,shape);
		self.cg_order[0].append_nodes(vec![sn]);
		self.cg_order[0].append_neurons(vec![Rc::clone(&output)]);
		output
	}

	pub fn transpose(&mut self, x:NNNeuron<T>) -> NNNeuron<T> {
		let (sn,output) = SynapseNode::<T>::transpose(x);
		self.cg_order[0].append_nodes(vec![sn]);
		self.cg_order[0].append_neurons(vec![Rc::clone(&output)]);
		output
	}

	pub fn sum_to(&mut self, x:NNNeuron<T>, shape:Vec<usize>) -> NNNeuron<T> {
		let (sn,output) = SynapseNode::<T>::sum_to(x, &shape);
		self.cg_order[0].append_nodes(vec![sn]);
		self.cg_order[0].append_neurons(vec![Rc::clone(&output)]);
		output
	}

	pub fn broadcast_to(&mut self, x:NNNeuron<T>, shape:Vec<usize>) -> NNNeuron<T> {
		let (sn,output) = SynapseNode::<T>::broadcast_to(x, &shape);
		self.cg_order[0].append_nodes(vec![sn]);
		self.cg_order[0].append_neurons(vec![Rc::clone(&output)]);
		output
	}

	pub fn add(&mut self, x:NNNeuron<T>, y:NNNeuron<T>) -> NNNeuron<T> {
		let (sn,output) = SynapseNode::<T>::add(x,y);
		self.cg_order[0].append_nodes(vec![sn]);
		self.cg_order[0].append_neurons(vec![Rc::clone(&output)]);
		output
	}

	pub fn sub(&mut self, x:NNNeuron<T>, y:NNNeuron<T>) -> NNNeuron<T> {
		let (sn,output) = SynapseNode::<T>::sub(x,y);
		self.cg_order[0].append_nodes(vec![sn]);
		self.cg_order[0].append_neurons(vec![Rc::clone(&output)]);
		output
	}

	pub fn mul_rank0(&mut self, x:NNNeuron<T>, y:NNNeuron<T>) -> NNNeuron<T> {
		let (sn,output) = SynapseNode::<T>::mul_rank0(x,y);
		self.cg_order[0].append_nodes(vec![sn]);
		self.cg_order[0].append_neurons(vec![Rc::clone(&output)]);
		output
	}

	pub fn div_rank0(&mut self, x:NNNeuron<T>, y:NNNeuron<T>) -> NNNeuron<T> {
		let (sn,output) = SynapseNode::<T>::div_rank0(x,y);
		self.cg_order[0].append_nodes(vec![sn]);
		self.cg_order[0].append_neurons(vec![Rc::clone(&output)]);
		output
	}

	pub fn pow_rank0(&mut self, a:NNNeuron<T>, x:NNNeuron<T>) -> NNNeuron<T> {
		let (sn,output) = SynapseNode::<T>::pow_rank0(a,x);
		self.cg_order[0].append_nodes(vec![sn]);
		self.cg_order[0].append_neurons(vec![Rc::clone(&output)]);
		output
	}

	pub fn pow(&mut self, a:NNNeuron<T>, x:NNNeuron<T>) -> NNNeuron<T> {
		let (sn,output) = SynapseNode::<T>::pow(a,x);
		self.cg_order[0].append_nodes(vec![sn]);
		self.cg_order[0].append_neurons(vec![Rc::clone(&output)]);
		output
	}

	pub fn hadamard_product(&mut self, x:NNNeuron<T>, y:NNNeuron<T>) -> NNNeuron<T> {
		let (sn,output) = SynapseNode::<T>::hadamard_product(x,y);
		self.cg_order[0].append_nodes(vec![sn]);
		self.cg_order[0].append_neurons(vec![Rc::clone(&output)]);
		output
	}

	pub fn hadamard_division(&mut self, x:NNNeuron<T>, y:NNNeuron<T>) -> NNNeuron<T> {
		let (sn,output) = SynapseNode::<T>::hadamard_division(x,y);
		self.cg_order[0].append_nodes(vec![sn]);
		self.cg_order[0].append_neurons(vec![Rc::clone(&output)]);
		output
	}

	pub fn matrix_product(&mut self, x:NNNeuron<T>, y:NNNeuron<T>) -> NNNeuron<T> {
		let (sn,output) = SynapseNode::<T>::matrix_product(x,y);
		self.cg_order[0].append_nodes(vec![sn]);
		self.cg_order[0].append_neurons(vec![Rc::clone(&output)]);
		output
	}

	pub fn sin(&mut self, x:NNNeuron<T>) -> NNNeuron<T> {
		let (sn,output) = SynapseNode::<T>::sin(x);
		self.cg_order[0].append_nodes(vec![sn]);
		self.cg_order[0].append_neurons(vec![Rc::clone(&output)]);
		output
	}

	pub fn cos(&mut self, x:NNNeuron<T>) -> NNNeuron<T> {
		let (sn,output) = SynapseNode::<T>::cos(x);
		self.cg_order[0].append_nodes(vec![sn]);
		self.cg_order[0].append_neurons(vec![Rc::clone(&output)]);
		output
	}

	pub fn tanh(&mut self, x:NNNeuron<T>) -> NNNeuron<T> {
		let (sn,output) = SynapseNode::<T>::tanh(x);
		self.cg_order[0].append_nodes(vec![sn]);
		self.cg_order[0].append_neurons(vec![Rc::clone(&output)]);
		output
	}

	pub fn sigmod(&mut self, x:NNNeuron<T>) -> NNNeuron<T> {
		let (sn,output) = SynapseNode::<T>::sigmod(x);
		self.cg_order[0].append_nodes(vec![sn]);
		self.cg_order[0].append_neurons(vec![Rc::clone(&output)]);
		output
	}

	
	pub fn mean_square_error(&mut self, x0:NNNeuron<T>, x1:NNNeuron<T>) -> NNNeuron<T> {
		let (sn,output) = SynapseNode::<T>::mean_square_error(x0, x1);
		self.cg_order[0].append_nodes(vec![sn]);
		self.cg_order[0].append_neurons(vec![Rc::clone(&output)]);
		output
	}

	pub fn forward_propagating(&mut self, order:usize) -> Result<Vec<NNNeuron<T>>,String> {
		if order < self.cg_order.len() {
			let outputs = self.cg_order[order].forward();
			Ok(outputs)
		}
		else {
			Err("invalid order".to_string())
		}
	}

	pub fn backward_propagating(&mut self, order:usize) -> Result<Vec<NNNeuron<T>>,String> {
		if order < self.cg_order.len() {
			let cg = self.cg_order[order].backward();
			if self.cg_order.len() <= order+1 {
				self.cg_order.push(cg);
			}
			else {
				self.cg_order[order+1] = cg;
			}

			let outputs = self.cg_order[order+1].forward();

			Ok(outputs)
		}
		else {
			Err("invalid order".to_string())
		}
	}
	pub fn clear_grads(&mut self, order:usize) -> Result<(),String>{
		if order < self.cg_order.len() {
			self.cg_order[order].clear_grads();
			Ok(())
		}
		else {
			Err("invalid order".to_string())
		}
	}

	pub fn make_dot_graph(&mut self, order:usize, file_name:&str) -> Result<(),String> {
		if order < self.cg_order.len() {
			self.cg_order[order].make_dot_graph(file_name, true);
			Ok(())
		}
		else {
			Err("invalid order".to_string())
		}
	}

}

