/* -*- tab-width:4 -*- */
use std::fmt;
use std::cell::{Ref,RefCell};
use std::rc::Rc;
use std::collections::HashMap;
use linear_transform::tensor::tensor_base::Tensor;

use crate::neuron::{NNNeuron,Neuron};
use crate::neuron::nn_neuron_new;

type MakeDiffNode<T> = fn (inputs: &Vec<NNNeuron<T>>, grads: &Vec<NNNeuron<T>>) -> Vec<NNSynapseNode<T>>;

pub struct Synapse<T>
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

pub struct SynapseNode<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone {
	name: String,
	inputs: Vec<NNNeuron<T>>,
	outputs: Vec<NNNeuron<T>>,
	synapse: Synapse<T>,
	generation: usize
}

pub type NNSynapseNode<T> = Rc<RefCell<SynapseNode<T>>>;

impl<T> fmt::Display for SynapseNode<T>
where T: fmt::Display + Clone + num::Float + num::pow::Pow<T, Output = T> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		let mut disp = format!("SynapseNode. name:{}\n",self.name);
		disp = format!("{}generation:{}", disp, self.generation);
		write!(f,"{}",disp)
	}
}

impl<T> SynapseNode<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone + fmt::Display {

	pub fn new(name:&str,
			   inputs: Vec<NNNeuron<T>>,
			   outputs: Vec<NNNeuron<T>>,
			   synapse: Synapse<T>) -> SynapseNode<T> {
		let input_generation:usize =
			inputs.iter().fold(0, |g_max,ni| {
				//println!("new sn: {},ni name={}", name, ni.borrow().name());
				let g = if let Some(generator) = ni.borrow().ref_generator() {
					generator.borrow().get_generation()+1
				}
				else {
					0
				};
				if g_max < g { g } else { g_max }
			});
		SynapseNode {
			name: name.to_string(),
			inputs,
			outputs,
			synapse,
			generation: input_generation
		}
	}

	pub fn get_generation(&self) -> usize {
		self.generation
	}

	pub fn set_generation(&mut self, generation:usize) {
		self.generation = generation
	}

	pub fn neg(x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "-".to_string()+x.borrow().name();
		let output = Rc::new(RefCell::new(Neuron::<T>::new(&label, Tensor::<T>::zero(&[1,1]))));
		let s = SynapseNode::<T>::new(&label,
									  vec![Rc::clone(&x)],
									  vec![Rc::clone(&output)],
									  Synapse {
										  forward: |inputs| {
											  vec![inputs[0].neg()]
										  },
										  make_diff_node: |inputs, grads| {
											  let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
											  for ni in inputs.iter() {
												  let mut n = ni.borrow_mut();
												  if let Some(ref g) = n.ref_grad() {
													  let (sn,output) = Self::neg(Rc::clone(&grads[0]));
													  sns.push(sn);
													  let (sn,output) = Self::add(Rc::clone(&g), output);
													  sns.push(sn);
													  n.set_grad(Rc::clone(&output));
												  }
												  else {
													  let (sn,output) = Self::neg(Rc::clone(&grads[0]));
													  sns.push(sn);
													  n.set_grad(Rc::clone(&output));
												  }
											  }
											  sns
										  }
									  });
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
	}

	pub fn add(x:NNNeuron<T>, y:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
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
											  let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
											  for ni in inputs.iter() {
												  let mut n = ni.borrow_mut();
												  if let Some(ref g) = n.ref_grad() {
													  let (sn, output) = Self::add(Rc::clone(&g), Rc::clone(&grads[0]));
													  n.set_grad(Rc::clone(&output));
													  sns.push(Rc::clone(&sn))
												  }
												  else {
													  n.set_grad(Rc::clone(&grads[0]))
												  }
											  }
											  sns
										  }
									  });
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
	}

	pub fn sub(x:NNNeuron<T>, y:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "(".to_string()+x.borrow().name() + "-" + y.borrow().name() +")";
		let output = Rc::new(RefCell::new(Neuron::<T>::new(&label, Tensor::<T>::zero(&[1,1]))));
		let s = SynapseNode::<T>::new(&label,
									  vec![Rc::clone(&x),Rc::clone(&y)],
									  vec![Rc::clone(&output)],
									  Synapse {
										  forward: |inputs| {
											  vec![(inputs[0] - inputs[1])]
										  },
										  make_diff_node: |inputs, grads| {
											  let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
											  let (sn,output) = Self::neg(Rc::clone(&grads[0]));
											  let mut n = inputs[0].borrow_mut();
											  if let Some(ref g) = n.ref_grad() {
												  let (sn, output) = Self::add(Rc::clone(&g), Rc::clone(&grads[0]));
												  n.set_grad(output);
												  sns.push(sn)
											  }
											  else {
												  n.set_grad(Rc::clone(&grads[0]));
											  }
											  sns.push(sn);
											  let mut n = inputs[1].borrow_mut();
											  if let Some(ref g) = n.ref_grad() {
												  let (sn, output) = Self::add(Rc::clone(&g), output);
												  n.set_grad(output);
												  sns.push(sn)
											  }
											  else {
												  n.set_grad(output);
											  }
											  sns
										  }
									  });
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
	}

	pub fn mul_rank0(x:NNNeuron<T>, y:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		assert_eq!(x.borrow().shape(),&[1,1]);
		assert_eq!(y.borrow().shape(),&[1,1]);
		let label = "(".to_string()+x.borrow().name() + "<^0*^0>" + y.borrow().name() +")";
		let output = Rc::new(RefCell::new(Neuron::<T>::new(&label, Tensor::<T>::zero(&[1,1]))));
		let s = SynapseNode::<T>::new(&label,
									  vec![Rc::clone(&x),Rc::clone(&y)],
									  vec![Rc::clone(&output)],
									  Synapse {
										  forward: |inputs| {
											  vec![Tensor::<T>::mul_rank0(inputs[0], inputs[1])]
										  },
										  make_diff_node: |inputs, grads| {
											  let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
											  let (l_sn, l_output) = Self::mul_rank0(Rc::clone(&grads[0]), Rc::clone(&inputs[1]));
											  let (r_sn, r_output) = Self::mul_rank0(Rc::clone(&grads[0]), Rc::clone(&inputs[0]));
											  sns.push(l_sn);
											  sns.push(r_sn);
											  {
												  let mut n = inputs[0].borrow_mut();
												  if let Some(ref g) = n.ref_grad() {
													  let (sn, output) = Self::add(Rc::clone(&g), l_output);
													  n.set_grad(output);
													  sns.push(sn)
												  }
												  else {
													  n.set_grad(l_output);
												  }
											  }
											  {
												  let mut n = inputs[1].borrow_mut();
												  if let Some(ref g) = n.ref_grad() {
													  let (sn, output) = Self::add(Rc::clone(&g), r_output);
													  n.set_grad(output);
													  sns.push(sn)
												  }
												  else {
													  n.set_grad(r_output);
												  }
											  }
											  sns
										  }
									  });
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
	}

	pub fn div_rank0(x:NNNeuron<T>, y:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		assert_eq!(x.borrow().shape(),&[1,1]);
		assert_eq!(y.borrow().shape(),&[1,1]);
		let label = "(".to_string()+x.borrow().name() + "<^0*^0>" + y.borrow().name() +")";
		let output = Rc::new(RefCell::new(Neuron::<T>::new(&label, Tensor::<T>::zero(&[1,1]))));
		let s = SynapseNode::<T>::new(&label,
									  vec![Rc::clone(&x),Rc::clone(&y)],
									  vec![Rc::clone(&output)],
									  Synapse {
										  forward: |inputs| {
											  vec![Tensor::<T>::mul_rank0(inputs[0], inputs[1])]
										  },
										  make_diff_node: |inputs, grads| {
											  let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
											  let two = Rc::new(RefCell::new(Neuron::<T>::new("2.0",
																							  (Tensor::<T>::one(&[1,1]))+Tensor::<T>::one(&[1,1]))));
											  let (l_sn, l_output) = Self::div_rank0(Rc::clone(&grads[0]), Rc::clone(&inputs[1]));
											  sns.push(l_sn);
											  let (neg, neg_input0) = Self::neg(Rc::clone(&inputs[0]));
											  sns.push(neg);
											  let (r_sn, r_output) = Self::pow_rank0(Rc::clone(&inputs[1]),two);
											  sns.push(r_sn);
											  let (r_sn, r_output) = Self::div_rank0(neg_input0, r_output);
											  sns.push(r_sn);
											  let (r_sn, r_output) = Self::mul_rank0(Rc::clone(&grads[0]), r_output);
											  sns.push(r_sn);
											  {
												  let mut n = inputs[0].borrow_mut();
												  if let Some(ref g) = n.ref_grad() {
													  let (sn, output) = Self::add(Rc::clone(&g), l_output);
													  n.set_grad(output);
													  sns.push(sn)
												  }
												  else {
													  n.set_grad(l_output);
												  }
											  }
											  {
												  let mut n = inputs[1].borrow_mut();
												  if let Some(ref g) = n.ref_grad() {
													  let (sn, output) = Self::add(Rc::clone(&g), r_output);
													  n.set_grad(output);
													  sns.push(sn)
												  }
												  else {
													  n.set_grad(r_output);
												  }
											  }
											  sns
										  }
									  });
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
	}

	fn exp(x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "exp^".to_string() + x.borrow().name();
		let output = Rc::new(RefCell::new(Neuron::<T>::new(&label, Tensor::<T>::zero(&[1,1]))));

		let s = SynapseNode::<T>::new(&label,
									  vec![Rc::clone(&x)],
									  vec![Rc::clone(&output)],
									  Synapse {
										  forward: |inputs| {
											  inputs.iter().map(|i| i.exp()).collect()
										  },
										  make_diff_node: |inputs, grads| {
											  let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
											  let (sn1,output) = Self::exp(Rc::clone(&inputs[0]));
											  sns.push(sn1);
											  let (sn2,output) = Self::mul_rank0(output,Rc::clone(&grads[0]));
											  sns.push(sn2);
											  {
												  let mut n = inputs[0].borrow_mut();
												  if let Some(ref g) = n.ref_grad() {
													  let (sn, output) = Self::add(Rc::clone(&g), output);
													  n.set_grad(output);
													  sns.push(sn)
												  }
												  else {
													  n.set_grad(output);
												  }
											  }
											  sns
										  }
									  });
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
    }

	fn pow_rank0(a:NNNeuron<T>, x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		assert_eq!(x.borrow().shape(),&[1,1]);
		assert_eq!(a.borrow().shape(),&[1,1]);
		let label = "(".to_string() + a.borrow().name() + "(^0)" + x.borrow().name() + x.borrow().name() + ")";
		let output = Rc::new(RefCell::new(Neuron::<T>::new(&label, Tensor::<T>::zero(&[1,1]))));
		let s = SynapseNode::<T>::new(&label,
									  vec![Rc::clone(&a),Rc::clone(&x)],
									  vec![Rc::clone(&output)],
									  Synapse {
										  forward: |inputs| {
											  vec![inputs[0].pow_rank0(inputs[1][vec![0,0]])]
										  },
										  make_diff_node: |inputs, grads| {
											  let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
											  let one = Rc::new(RefCell::new(Neuron::<T>::new("1.0", Tensor::<T>::one(&[1,1]))));
											  let (dec_index_sn, dec_index) = Self::sub(Rc::clone(&inputs[1]),one);
											  sns.push(dec_index_sn);
											  let (l_sn,l_output) = Self::pow_rank0(Rc::clone(&inputs[0]), dec_index);
											  sns.push(l_sn);
											  let (l_sn,l_output) = Self::mul_rank0(Rc::clone(&inputs[1]), l_output);
											  sns.push(l_sn);
											  let (l_sn,l_output) = Self::mul_rank0(Rc::clone(&grads[0]),  l_output);
											  sns.push(l_sn);
											  let (r_sn,r_output) = Self::pow_rank0(Rc::clone(&inputs[0]),Rc::clone(&inputs[1]));
											  sns.push(r_sn);
											  let (baselog_sn, baselog) = Self::ln_rank0(Rc::clone(&inputs[0]));
											  sns.push(baselog_sn);
											  let (r_sn,r_output) = Self::mul_rank0(baselog,r_output);
											  sns.push(r_sn);
											  {
												  let mut n = inputs[0].borrow_mut();
												  if let Some(ref g) = n.ref_grad() {
													  let (sn, output) = Self::add(Rc::clone(&g), l_output);
													  n.set_grad(output);
													  sns.push(sn)
												  }
												  else {
													  n.set_grad(l_output);
												  }
											  }
											  {
												  let mut n = inputs[1].borrow_mut();
												  if let Some(ref g) = n.ref_grad() {
													  let (sn, output) = Self::add(Rc::clone(&g), r_output);
													  n.set_grad(output);
													  sns.push(sn)
												  }
												  else {
													  n.set_grad(r_output);
												  }
											  }
											  sns
										  }
									  });
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
	}

	fn ln_rank0(x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "(ln0(".to_string() + x.borrow().name() + ")";
		let output = nn_neuron_new::<T>(&label, Tensor::<T>::zero(&[1,1]));
		let s = SynapseNode::<T>::new(&label,
									  vec![Rc::clone(&x)],
									  vec![Rc::clone(&output)],
									  Synapse {
										  forward: |inputs| {
											  vec![inputs[0].ln()]
										  },
										  make_diff_node: |inputs, grads| {
											  let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
											  // (ln x)' = 1/x
											  let one = Rc::new(RefCell::new(Neuron::<T>::new("1.0", Tensor::<T>::one(&[1,1]))));
											  let (inverse_sn, inverse) = Self::div_rank0(one, Rc::clone(&inputs[0]));
											  sns.push(inverse_sn);
											  let (sn, output) = Self::mul_rank0(Rc::clone(&grads[0]), inverse);
											  sns.push(sn);
											  {
												  let mut n = inputs[0].borrow_mut();
												  if let Some(ref g) = n.ref_grad() {
													  let (sn, output) = Self::add(Rc::clone(&g), output);
													  n.set_grad(output);
													  sns.push(sn)
												  }
												  else {
													  n.set_grad(output);
												  }
											  }
											  sns
										  }
									  });
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
	}

    fn forward(&self) -> Vec<NNNeuron<T>> {
		let inputs_holder = self.inputs.iter().map(|n| n.borrow()).collect::<Vec<Ref<'_,Neuron<T>>>>();
		let inputs = inputs_holder.iter().map(|n| n.ref_signal()).collect::<Vec<&Tensor<T>>>();
		let outputs = (self.synapse.forward)(inputs);
		self.outputs.iter().zip(outputs.into_iter()).map(|(n,t)| {n.borrow_mut().assign(t); Rc::clone(n)}).collect()
    }

    fn make_diff_node(&self) -> Vec<NNSynapseNode<T>> {
		let grads =
			self.outputs.iter().map(|no| no.borrow_mut().get_grad()).collect();
		(self.synapse.make_diff_node)(&self.inputs, &grads)
    }

}

struct NeuralNetwork<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone {
	neurons: HashMap<* const RefCell<Neuron<T>>, NNNeuron<T>>,
	synapse_node: HashMap<* const RefCell<SynapseNode<T>>, Rc<RefCell<SynapseNode<T>>>>,
	generation_table: Vec<NNSynapseNode<T>>
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
			let sns = add1.borrow().make_diff_node();
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
				let (sq_exp_sq_sn, y) = SynapseNode::<f64>::pow_rank0(b,two);

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
		}
	}

}
