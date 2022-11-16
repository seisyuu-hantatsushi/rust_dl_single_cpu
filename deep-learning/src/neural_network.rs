/* -*- tab-width:4 -*- */
use std::{fmt,fs};
use std::io::Write;
use std::cell::RefCell;
use std::rc::Rc;
use std::collections::{BTreeSet,HashMap};
use linear_transform::tensor::Tensor;

use crate::neuron::{NNNeuron,Neuron,nn_neuron_new,nn_neuron_constant};
use crate::synapse::{NNSynapseNode,SynapseNode};

struct ComputationalGraph<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone + fmt::Display {
	neurons: HashMap<* const RefCell<Neuron<T>>, NNNeuron<T>>,
	synapse_nodes: HashMap<* const RefCell<SynapseNode<T>>, Rc<RefCell<SynapseNode<T>>>>,
	generation_table: Vec<Vec<NNSynapseNode<T>>>
}

impl<T> ComputationalGraph<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone + fmt::Display {

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
where T:num::Float + num::pow::Pow<T, Output = T> + Clone + fmt::Display {
	cg_order: Vec<ComputationalGraph<T>>
}

impl<T> NeuralNetwork<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone + fmt::Display {
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

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn nn_test() {
		{
			let x = nn_neuron_new::<f64>("x", Tensor::<f64>::from_array(&[1,1],&[3.0]));
			let (add1,output) = SynapseNode::<f64>::add(Rc::clone(&x),Rc::clone(&x));
			let (add2,output) = SynapseNode::<f64>::add(Rc::clone(&output),Rc::clone(&x));
			add2.borrow().make_diff_node();
			let (sns,_outputs) = add1.borrow().make_diff_node();
			sns[0].borrow().forward();
			let gx = sns[1].borrow().forward();
			assert_eq!(output.borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1],&[9.0]));
			assert_eq!(gx[0].borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1],&[3.0]));
		}

		{
			let x = nn_neuron_new::<f64>("x", Tensor::<f64>::from_array(&[1,1],&[2.0]));
			let (neg, output) = SynapseNode::<f64>::neg(Rc::clone(&x));
			let (sub1,output) = SynapseNode::<f64>::sub(output,Rc::clone(&x));
			let (sub2,output) = SynapseNode::<f64>::sub(output,Rc::clone(&x));
			sub2.borrow().make_diff_node();
			sub1.borrow().make_diff_node();
			neg.borrow().make_diff_node();

			assert_eq!(output.borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1],&[-6.0]));

			let borrowed_x = x.borrow();
			if let Some(ref g) = borrowed_x.ref_grad() {
				assert_eq!(g.borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1],&[-3.0]));
			}
			else {
				println!("x.grad=None");
				assert!(false);
			}

			let y = nn_neuron_new::<f64>("y", Tensor::<f64>::from_array(&[1,1],&[2.0]));
			let (neg, neg_y) = SynapseNode::<f64>::neg(Rc::clone(&y));
			let (add1,output) = SynapseNode::<f64>::add(Rc::clone(&neg_y),Rc::clone(&neg_y));
			let (add2,output) = SynapseNode::<f64>::add(output,Rc::clone(&neg_y));
			assert_eq!(output.borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1],&[-6.0]));
			add2.borrow().make_diff_node();
			add1.borrow().make_diff_node();
			neg.borrow().make_diff_node();
			let borrowed_y = y.borrow();
			if let Some(ref g) = borrowed_y.ref_grad() {
				assert_eq!(g.borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1],&[-3.0]));
			}
			else {
				println!("y.grad=None");
				assert!(false);
			}
		}

		{
			let a = nn_neuron_new::<f64>("a", Tensor::<f64>::from_array(&[1,1],&[3.0]));
			let b = nn_neuron_new::<f64>("b", Tensor::<f64>::from_array(&[1,1],&[2.0]));
			let c = nn_neuron_new::<f64>("c", Tensor::<f64>::from_array(&[1,1],&[1.0]));
			let (mul1,output) = SynapseNode::<f64>::mul_rank0(Rc::clone(&a),Rc::clone(&b));
			let (add1,output) = SynapseNode::<f64>::add(output,c);
			assert_eq!(output.borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1],&[7.0]));
			add1.borrow().make_diff_node();
			mul1.borrow().make_diff_node();

			{
				let borrowed_a = a.borrow();
				if let Some(ref g) = borrowed_a.ref_grad() {
					assert_eq!(g.borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1],&[2.0]));
				}
				else {
					println!("a.grad=None");
					assert!(false);
				}
			}
			{
				let borrowed_b = b.borrow();
				if let Some(ref g) = borrowed_b.ref_grad() {
					assert_eq!(g.borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1],&[3.0]));
				}
				else {
					println!("b.grad=None");
					assert!(false);
				}
			}
		}

		{
			let x = 3.0;
			let a = nn_neuron_new::<f64>("a", Tensor::<f64>::from_array(&[1,1],&[x]));
			let (mul1,output) = SynapseNode::<f64>::mul_rank0(Rc::clone(&a),Rc::clone(&a));
			/* z = a*a */
			println!("{}",output.borrow());
			mul1.borrow().make_diff_node();
			{
				let borrowed_a = a.borrow();
				if let Some(ref g) = borrowed_a.ref_grad() {
					/*
					fn square(x:f64) -> f64 {
						x*x
					}
					let delta = 1.0e-4;
					let diff = (square(x+delta)-square(x-delta))/(2.0*delta);
					println!("{}", diff);
					 */
					assert_eq!(g.borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1],&[6.0]));
				}
				else {
					println!("a.grad=None");
					assert!(false);
				}
			}

			{
				let x0 = 0.5;
				let x = nn_neuron_new::<f64>("a", Tensor::<f64>::from_array(&[1,1],&[x0]));
				let two =  nn_neuron_new::<f64>("2.0", Tensor::<f64>::from_array(&[1,1],&[2.0]));
				let (square_sn, a) = SynapseNode::<f64>::pow_rank0(Rc::clone(&x),Rc::clone(&two));
				let (exp_sq_sn, b) = SynapseNode::<f64>::exp(a);
				let (sq_exp_sq_sn, _y) = SynapseNode::<f64>::pow_rank0(b,two);

				fn func(x:f64) -> f64 {
					x.powf(2.0).exp().powf(2.0)
				}
				sq_exp_sq_sn.borrow().make_diff_node();
				exp_sq_sn.borrow().make_diff_node();
				square_sn.borrow().make_diff_node();
				let borrowed_x = x.borrow();
				if let Some(ref g) = borrowed_x.ref_grad() {
					let delta =  1.0e-4;
					let diff = (func(x0+delta)-func(x0-delta))/(2.0*delta);
					let dx = g.borrow().ref_signal()[vec![0,0]];
					assert!((diff-dx).abs() < delta);
				}
				else {
					println!("a.grad=None");
					assert!(false);
				}
			}

			{
				let mut nn = NeuralNetwork::<f64>::new();
				let (x0,y0) = (1.0,1.0);
				let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[1,1],&[x0]));
				let y = nn.create_neuron("y", Tensor::<f64>::from_array(&[1,1],&[y0]));
				let z = nn.add(x,y);
				z.borrow_mut().rename("z");
				println!("{}", z.borrow());
				if let Err(e) = nn.make_dot_graph(0,"order0_graph.dot") {
					println!("{}",e);
					assert!(false)
				}
			}

			{
				let mut nn = NeuralNetwork::<f64>::new();
				let x0 = 2.0;
				let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[1,1],&[x0]));
				let c4 = nn.create_constant("4.0", Tensor::<f64>::from_array(&[1,1],&[4.0]));
				let y = nn.pow_rank0(Rc::clone(&x),c4);
				y.borrow_mut().rename("y");

				println!("{}", y.borrow());

				let _ = nn.clear_grads(0);
				match nn.backward_propagating(0) {
					Ok(outputs) => {
						for output in outputs.iter() {
							println!("gx1 {}",output.borrow());
						}
					},
					Err(e) => {
						println!("{}",e);
						assert!(false)
					}
				}

				let _ = nn.clear_grads(0);
				let _ = nn.clear_grads(1);
				match nn.backward_propagating(1) {
					Ok(outputs) => {
						for output in outputs.iter() {
							println!("gx2 {}",output.borrow());
						}
					},
					Err(e) => {
						println!("{}",e);
						assert!(false)
					}
				}

				let _ = nn.clear_grads(0);
				let _ = nn.clear_grads(1);
				let _ = nn.clear_grads(2);
				match nn.backward_propagating(2) {
					Ok(outputs) => {
						for output in outputs.iter() {
							println!("gx3 {}",output.borrow());
						}
					},
					Err(e) => {
						println!("{}",e);
						assert!(false)
					}
				}

				let _ = nn.clear_grads(0);
				let _ = nn.clear_grads(1);
				let _ = nn.clear_grads(2);
				let _ = nn.clear_grads(3);
				match nn.backward_propagating(3) {
					Ok(outputs) => {
						for output in outputs.iter() {
							println!("gx4 {}",output.borrow());
						}
					},
					Err(e) => {
						println!("{}",e);
						assert!(false)
					}
				}

				let _ = nn.clear_grads(0);
				let _ = nn.clear_grads(1);
				let _ = nn.clear_grads(2);
				let _ = nn.clear_grads(3);
				let _ = nn.clear_grads(4);
				match nn.backward_propagating(4) {
					Ok(outputs) => {
						for output in outputs.iter() {
							println!("gx5 {}",output.borrow());
						}
					},
					Err(e) => {
						println!("{}",e);
						assert!(false)
					}
				}

				let _ = nn.clear_grads(0);
				let _ = nn.clear_grads(1);
				let _ = nn.clear_grads(2);
				let _ = nn.clear_grads(3);
				let _ = nn.clear_grads(4);
				let _ = nn.clear_grads(5);
				match nn.backward_propagating(5) {
					Ok(outputs) => {
						for output in outputs.iter() {
							println!("gx6 {}",output.borrow());
						}
					},
					Err(e) => {
						println!("{}",e);
						assert!(false)
					}
				}

				let _ = nn.clear_grads(0);
				let _ = nn.clear_grads(1);
				let _ = nn.clear_grads(2);
				let _ = nn.clear_grads(3);
				let _ = nn.clear_grads(4);
				let _ = nn.clear_grads(5);
				let _ = nn.clear_grads(6);
				match nn.backward_propagating(6) {
					Ok(outputs) => {
						for output in outputs.iter() {
							println!("gx7 {}",output.borrow());
						}
					},
					Err(e) => {
						println!("{}",e);
						assert!(false)
					}
				}

				if let Err(e) = nn.make_dot_graph(0,"graph2_order0.dot") {
					println!("{}",e);
					assert!(false)
				}

				if let Err(e) = nn.make_dot_graph(1,"graph2_order1.dot") {
					println!("{}",e);
					assert!(false)
				}

				if let Err(e) = nn.make_dot_graph(2,"graph2_order2.dot") {
					println!("{}",e);
					assert!(false)
				}

				if let Err(e) = nn.make_dot_graph(3,"graph2_order3.dot") {
					println!("{}",e);
					assert!(false)
				}

				if let Err(e) = nn.make_dot_graph(4,"graph2_order4.dot") {
					println!("{}",e);
					assert!(false)
				}
			}

			{
				let mut nn = NeuralNetwork::<f64>::new();
				let x0 = 2.0;
				let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[1,1],&[x0]));
				let c4 = nn.create_constant("4.0", Tensor::<f64>::from_array(&[1,1],&[4.0]));
				let c2 = nn.create_constant("2.0", Tensor::<f64>::from_array(&[1,1],&[2.0]));
				let term1 = nn.pow_rank0(Rc::clone(&x),c4);
				let term2 = nn.pow_rank0(Rc::clone(&x),Rc::clone(&c2));
				let term3 = nn.mul_rank0(Rc::clone(&c2), term2);
				let y = nn.sub(term1,term3);
				y.borrow_mut().rename("y");

				if let Err(e) = nn.make_dot_graph(0,"graph2.dot") {
					println!("{}",e);
					assert!(false)
				}
				fn func(x:f64) -> f64 {
					x.powf(4.0) - 2.0 * x.powf(2.0)
				}
				assert_eq!(y.borrow().ref_signal()[vec![0,0]],func(x0));

				match nn.backward_propagating(0) {
					Ok(outputs) => {
						for output in outputs.iter() {
							let delta = 1.0e-5;
							let diff = (func(x0+delta)-func(x0-delta))/(2.0*delta);
							println!("gx1 {}",output.borrow());
							let gx = output.borrow().ref_signal()[vec![0,0]];
							assert!((diff-gx).abs() <= delta);
						}
					},
					Err(e) => {
						println!("{}",e);
						assert!(false)
					}
				}

				let _ = nn.clear_grads(0);
				let _ = nn.clear_grads(1);

				match nn.backward_propagating(1) {
					Ok(_outputs) => {
						let borrowed_x = x.borrow();
						if let Some(ref g) = borrowed_x.ref_grad() {
							let delta = 1.0e-5;
							let diff2 = (func(x0+delta) + func(x0-delta) - 2.0 * func(x0))/(delta.powf(2.0));
							println!("gx2 {}",g.borrow());
							let gx2 = g.borrow().ref_signal()[vec![0,0]];
							assert!((diff2-gx2).abs() <= delta);
						}
						else {
							assert!(false);
						}
					},
					Err(e) => {
						println!("{}",e);
						assert!(false)
					}
				}

				if let Err(e) = nn.make_dot_graph(0,"graph3_order0.dot") {
					println!("{}",e);
					assert!(false)
				}

				if let Err(e) = nn.make_dot_graph(1,"graph3_order1.dot") {
					println!("{}",e);
					assert!(false)
				}

				if let Err(e) = nn.make_dot_graph(2,"graph3_order2.dot") {
					println!("{}",e);
					assert!(false)
				}
			}
		}

		{
			//Taylor Expansion of Sin
			let mut nn = NeuralNetwork::<f64>::new();
			let x0 = std::f64::consts::PI/2.0;
			let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[1,1],&[x0]));
			let mut tail_term = Rc::clone(&x);
			let mut counter = 2;
			loop {
				let index_int = (counter-1)*2 + 1;
				let index_real = index_int as f64;
				let index = nn.create_constant(&index_int.to_string(),
											   Tensor::<f64>::from_array(&[1,1],&[index_real]));
				let sign:f64 = (-1.0f64).powi(counter-1);
				let factorial:f64 = sign*(1..(index_int+1)).fold(1.0, |p, n| p * (n as f64));
				let p = nn.pow_rank0(Rc::clone(&x), index);
				let factorial_constant = nn.create_constant(&format!("{}({}!)",sign,index_int),
															Tensor::<f64>::from_array(&[1,1],&[factorial]));
				let term = nn.div_rank0(Rc::clone(&p), factorial_constant);
				let label = "term_".to_string() + &counter.to_string();
				term.borrow_mut().rename(&label);
				tail_term = nn.add(tail_term, Rc::clone(&term));

				let t = term.borrow().ref_signal()[vec![0,0]];
				if t.abs() <= 1.0e-6 {
					break;
				}
				counter += 1;
				if counter >= 10000 {
					break;
				}
			}
			{
				let diff = (tail_term.borrow().ref_signal()[vec![0,0]]-x0.sin()).abs();
				println!("{} {}",
						 tail_term.borrow().ref_signal()[vec![0,0]],
						 x0.sin());
				assert!(diff < 1.0e-5);
			}

			match nn.backward_propagating(0) {
				Ok(_outputs) => {
					let borrowed_x = x.borrow();
					if let Some(ref g) = borrowed_x.ref_grad() {
						let delta = 1.0e-5;
						let diff = ((x0+delta).sin() - (x0-delta).sin())/(2.0*delta);
						println!("gx {} {}",g.borrow(),diff);
					}
					else {
						assert!(false);
					}
				},
				Err(e) => {
					println!("{}",e);
					assert!(false)
				}
			}

			if let Err(e) = nn.make_dot_graph(0,"sin_taylor_order0.dot") {
				println!("{}",e);
				assert!(false)
			}

			if let Err(e) = nn.make_dot_graph(1,"sin_taylor_order1.dot") {
				println!("{}",e);
				assert!(false)
			}
		}
	}
}
