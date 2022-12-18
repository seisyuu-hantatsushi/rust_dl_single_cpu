/* -*- tab-width:4 -*- */
use std::fmt;
use std::cell::{Ref,RefCell};
use std::rc::Rc;
use linear_transform::tensor::Tensor;

use crate::neuron::{NeuronPrimType,NNNeuron,Neuron,nn_neuron_new,nn_neuron_constant};

pub enum SynapseOption {
	BroadcastTo((Vec<usize>,Vec<usize>)),
	Reshape((Vec<usize>,Vec<usize>)),
	Sum((Vec<usize>,Vec<usize>))
}

pub type ForwardProp<T> = fn (inputs: Vec<&Tensor<T>>, synapse_opt: &Option<SynapseOption>)
							  -> Vec<Tensor<T>>;
pub type MakeDiffNode<T> = fn (inputs: &Vec<NNNeuron<T>>,
							   grads: &Vec<NNNeuron<T>>,
							   synapse_opt: &Option<SynapseOption>)
							   -> (Vec<NNSynapseNode<T>>,Vec<NNNeuron<T>>);

pub struct Synapse<T>
where T:NeuronPrimType<T> {
	forward: ForwardProp<T>,
	make_diff_node: MakeDiffNode<T>,
	synapse_opt: Option<SynapseOption>
}

impl<T> Synapse<T>
where T:NeuronPrimType<T> {
	pub fn new(forward: ForwardProp<T>,
			   make_diff_node: MakeDiffNode<T> ) -> Synapse<T> {
		Synapse {
			forward,
			make_diff_node,
			synapse_opt: None
		}
	}
	pub fn new_with_option(forward: ForwardProp<T>,
						   make_diff_node: MakeDiffNode<T>,
						   opt: SynapseOption) -> Synapse<T> {
		Synapse {
			forward,
			make_diff_node,
			synapse_opt: Some(opt)
		}
	}
	pub fn ref_option(&self) -> &Option<SynapseOption> {
		&self.synapse_opt
	}
}

pub struct SynapseNode<T>
where T:NeuronPrimType<T> {
	name: String,
	inputs: Vec<NNNeuron<T>>,
	outputs: Vec<NNNeuron<T>>,
	synapse: Synapse<T>,
	generation: usize
}

pub type NNSynapseNode<T> = Rc<RefCell<SynapseNode<T>>>;

impl<T> fmt::Display for SynapseNode<T>
where T: NeuronPrimType<T> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		let mut disp = format!("SynapseNode. name:{}\n",self.name);
		disp = format!("{}generation:{}", disp, self.generation);
		write!(f,"{}",disp)
	}
}

impl<T> SynapseNode<T>
where T:NeuronPrimType<T> {

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

	pub fn name(&self) -> &str {
		&self.name
	}

	pub fn get_generation(&self) -> usize {
		self.generation
	}

	pub fn set_generation(&mut self, generation:usize) {
		self.generation = generation
	}

	pub fn inputs(&self) -> &Vec<Rc<RefCell<Neuron<T>>>> {
		&self.inputs
    }

	pub fn outputs(&self) -> &Vec<Rc<RefCell<Neuron<T>>>> {
		&self.outputs
	}

	pub fn neg(x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "neg";
		let output = Rc::new(RefCell::new(Neuron::<T>::new(&label, Tensor::<T>::zero(&[1,1]))));
		let s = SynapseNode::<T>::new(&label,
									  vec![Rc::clone(&x)],
									  vec![Rc::clone(&output)],
									  Synapse::<T>:: new(
										  |inputs, _opt| {
											  vec![inputs[0].neg()]
										  },
										  |inputs, grads, _opt| {
											  let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
											  let mut outputs:Vec<NNNeuron<T>> = Vec::new();
											  let mut linked_grad = false;
											  for ni in inputs.iter() {
												  let mut n = ni.borrow_mut();
												  if !n.is_constant() {
													  linked_grad = true;
													  let output = if grads[0].borrow().is_constant() {
														  let t = grads[0].borrow().ref_signal().neg();
														  let label = "-(".to_string() + grads[0].borrow().name() + ")";
														  nn_neuron_constant(&label, t)
													  }
													  else {
														  let (sn,output) = Self::neg(Rc::clone(&grads[0]));
														  sns.push(sn);
														  outputs.push(Rc::clone(&output));
														  output
													  };
													  if let Some(ref g) = n.ref_grad() {
														  let (sn,output) = Self::add(Rc::clone(&g), output);
														  sns.push(sn);
														  outputs.push(Rc::clone(&output));
														  n.set_grad(output);
													  }
													  else {
														  n.set_grad(output);
													  }
												  }
											  }
											  if linked_grad {
												  outputs.push(Rc::clone(&grads[0]))
											  }
											  (sns,outputs)
										  })
									  );
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
	}

	pub fn add(x:NNNeuron<T>, y:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "add";
		let output = Rc::new(RefCell::new(Neuron::<T>::new(&label, Tensor::<T>::zero(&[1,1]))));
		let s = Synapse::<T>::new(
			|inputs, _opt| {
				let left_shape  = inputs[0].shape().to_vec();
				let right_shape = inputs[1].shape().to_vec();
				let left_prod  = left_shape.iter().fold(1, |prod, &e| { prod * e });
				let right_prod = right_shape.iter().fold(1, |prod, &e| { prod * e });
				if left_prod < right_prod {
					let expand_left = inputs[0].broadcast(&right_shape);
					vec![expand_left + inputs[1]]
				}
				else if left_prod > right_prod {
					let expand_right = inputs[1].broadcast(&left_shape);
					vec![inputs[0] + expand_right]
				}
				else {
					vec![inputs[0] + inputs[1]]
				}
			},
			|inputs, grads, _opt| {
				let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
				let mut outputs:Vec<NNNeuron<T>> = Vec::new();
				let mut left_neuron  = inputs[0].borrow_mut();
				let mut right_neuron = inputs[1].borrow_mut();

				let left_shape  = left_neuron.ref_signal().shape();
				let right_shape = right_neuron.ref_signal().shape();

				if left_neuron.is_constant() && right_neuron.is_constant(){
					return (sns,outputs);
				}

				outputs.push(Rc::clone(&grads[0]));
				if !left_neuron.is_constant() {
					let grad = if grads[0].borrow().ref_signal().shape() != left_shape {
						let (sn, output) = Self::sum_to(Rc::clone(&grads[0]), left_shape);
						sns.push(sn);
						outputs.push(Rc::clone(&output));
						output
					}
					else {
						Rc::clone(&grads[0])
					};

					if let Some(ref g) = left_neuron.ref_grad() {
						outputs.push(Rc::clone(g));
						let (sn, output) = Self::add(Rc::clone(&g), grad);
						left_neuron.set_grad(Rc::clone(&output));
						sns.push(sn);
						outputs.push(output)
					}
					else {
						left_neuron.set_grad(grad)
					}
				}

				if !right_neuron.is_constant() {
					let grad = if grads[0].borrow().ref_signal().shape() != right_shape {
						let (sn, output) = Self::sum_to(Rc::clone(&grads[0]), right_shape);
						sns.push(sn);
						outputs.push(Rc::clone(&output));
						output
					}
					else {
						Rc::clone(&grads[0])
					};

					if let Some(ref g) = right_neuron.ref_grad() {
						outputs.push(Rc::clone(g));
						let (sn, output) = Self::add(Rc::clone(&g), grad);
						left_neuron.set_grad(Rc::clone(&output));
						sns.push(sn);
						outputs.push(output)
					}
					else {
						right_neuron.set_grad(grad)
					}
				}
				(sns,outputs)
			});
		let sn = SynapseNode::<T>::new(&label,
									   vec![Rc::clone(&x),Rc::clone(&y)],
									   vec![Rc::clone(&output)],
									   s);

		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn, output)
	}

	pub fn sub(x:NNNeuron<T>, y:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "sub";
		let output = Rc::new(RefCell::new(Neuron::<T>::new(&label, Tensor::<T>::zero(&[1,1]))));
		let s = Synapse::<T>::new(
			|inputs, _opt| {
				let left_shape = inputs[0].shape().to_vec();
				let right_shape = inputs[1].shape().to_vec();
				let left_prod  = left_shape.iter().fold(1, |prod, &e| { prod * e });
				let right_prod = right_shape.iter().fold(1, |prod, &e| { prod * e });
				if left_prod < right_prod {
					let expand_left = inputs[0].broadcast(&right_shape);
					vec![expand_left - inputs[1]]
				}
				else if left_prod > right_prod {
					let expand_right = inputs[1].broadcast(&left_shape);
					vec![inputs[0] - expand_right]
				}
				else {
					vec![inputs[0] - inputs[1]]
				}
			},
			|inputs, grads, _opt| {
				let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
				let mut outputs:Vec<NNNeuron<T>> = Vec::new();
				let mut left_neuron  = inputs[0].borrow_mut();
				let mut right_neuron = inputs[1].borrow_mut();

				let left_shape  = left_neuron.ref_signal().shape();
				let right_shape = right_neuron.ref_signal().shape();

				if left_neuron.is_constant() && right_neuron.is_constant() {
					return (sns,outputs);
				}

				outputs.push(Rc::clone(&grads[0]));
				if !left_neuron.is_constant() {
					let grad = if grads[0].borrow().ref_signal().shape() != left_shape {
						let (sn,output) = Self::sum_to(Rc::clone(&grads[0]), left_shape);
						sns.push(sn);
						outputs.push(Rc::clone(&output));
						output
					}
					else {
						Rc::clone(&grads[0])
					};

					if let Some(ref g) = left_neuron.ref_grad() {
						outputs.push(Rc::clone(g));
						let (sn, output) = Self::sub(Rc::clone(&g), grad);
						left_neuron.set_grad(Rc::clone(&output));
						sns.push(sn);
						outputs.push(output)
					}
					else {
						left_neuron.set_grad(grad)
					}
				}

				if !right_neuron.is_constant() {

					let (sn, neg_grad) = Self::neg(Rc::clone(&grads[0]));
					sns.push(sn);

					let grad = if neg_grad.borrow().ref_signal().shape() != right_shape {
						outputs.push(Rc::clone(&neg_grad));
						let (sn, output) = Self::sum_to(neg_grad, right_shape);
						sns.push(sn);
						outputs.push(Rc::clone(&output));
						output
					}
					else {
						neg_grad
					};
					outputs.push(Rc::clone(&grad));
					if let Some(ref g) = right_neuron.ref_grad() {
						outputs.push(Rc::clone(g));
						let (sn, output) = Self::add(Rc::clone(&g), grad);
						left_neuron.set_grad(Rc::clone(&output));
						sns.push(sn);
						outputs.push(output)
					}
					else {
						right_neuron.set_grad(grad)
					}
				}

				(sns,outputs)
			});
		let sn = SynapseNode::<T>::new(&label,
									   vec![Rc::clone(&x),Rc::clone(&y)],
									   vec![Rc::clone(&output)],
									   s);
		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn, output)
	}

	pub fn exp(x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "exp".to_string();
		let output = Rc::new(RefCell::new(Neuron::<T>::new(&label, Tensor::<T>::zero(&[1,1]))));

		let s = SynapseNode::<T>::new(&label,
									  vec![Rc::clone(&x)],
									  vec![Rc::clone(&output)],
									  Synapse::<T>::new(
										  |inputs, _opt| {
											  inputs.iter().map(|i| i.exp()).collect()
										  },
										  |inputs, grads, _opt| {
											  let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
											  let mut outputs:Vec<NNNeuron<T>> = Vec::new();
											  let (sn1,output) = Self::exp(Rc::clone(&inputs[0]));
											  sns.push(sn1);
											  outputs.push(Rc::clone(&output));
											  let (sn2,output) = Self::hadamard_product(output,Rc::clone(&grads[0]));
											  sns.push(sn2);
											  outputs.push(Rc::clone(&output));
											  {
												  let mut n = inputs[0].borrow_mut();
												  if let Some(ref g) = n.ref_grad() {
													  outputs.push(Rc::clone(&g));
													  let (sn, output) = Self::add(Rc::clone(&g), output);
													  n.set_grad(Rc::clone(&output));
													  sns.push(sn);
													  outputs.push(output);
												  }
												  else {
													  n.set_grad(output);
												  }
											  }
											  (sns,outputs)
										  }));
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
	}

	pub fn tanh(x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "tanh";
		let output = nn_neuron_new::<T>(&label, Tensor::<T>::zero(&[1,1]));

		let s = SynapseNode::<T>::new(&label,
									  vec![Rc::clone(&x)],
									  vec![Rc::clone(&output)],
									  Synapse::<T>::new(
										  |inputs,_opt| {
											  vec![inputs[0].tanh()]
										  },
										  |inputs,grads,_opt| {
											  let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
											  let mut outputs:Vec<NNNeuron<T>> = Vec::new();
											  if !inputs[0].borrow().is_constant() {
												  let (sn,output) = Self::tanh(Rc::clone(&inputs[0]));
												  sns.push(sn);
												  outputs.push(Rc::clone(&output));
												  outputs.push(Rc::clone(&inputs[0]));
												  let (sn,output) = Self::hadamard_product(Rc::clone(&output),output);
												  sns.push(sn);
												  outputs.push(Rc::clone(&output));
												  let one =
													  nn_neuron_constant("1.0", Tensor::<T>::one(grads[0].borrow().shape()));
												  outputs.push(Rc::clone(&one));
												  let (sn,output) = Self::sub(one, output);
												  sns.push(sn);
												  outputs.push(Rc::clone(&output));
												  let (sn,output) = Self::hadamard_product(Rc::clone(&grads[0]), output);
												  sns.push(sn);
												  outputs.push(Rc::clone(&grads[0]));
												  {
													  let mut n = inputs[0].borrow_mut();
													  let output = if let Some(ref g) = n.ref_grad() {
														  outputs.push(Rc::clone(&g));
														  let (sn, output) = Self::add(Rc::clone(&g), output);
														  n.set_grad(Rc::clone(&output));
														  sns.push(sn);
														  output
													  }
													  else {
														  n.set_grad(Rc::clone(&output));
														  output
													  };
													  outputs.push(output);
												  }
											  }
											  (sns,outputs)
										  }
									  ));

		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
	}

	pub fn forward(&self) -> Vec<NNNeuron<T>> {
		let inputs_holder = self.inputs.iter().map(|n| n.borrow()).collect::<Vec<Ref<'_,Neuron<T>>>>();
		let inputs = inputs_holder.iter().map(|n| n.ref_signal()).collect::<Vec<&Tensor<T>>>();
		let outputs = (self.synapse.forward)(inputs, self.synapse.ref_option());
		self.outputs.iter().zip(outputs.into_iter()).map(|(n,t)| {n.borrow_mut().assign(t); Rc::clone(n)}).collect()
    }

    pub fn make_diff_node(&self) -> (Vec<NNSynapseNode<T>>, Vec<NNNeuron<T>>) {
		let grads =
			self.outputs.iter().map(|no| no.borrow_mut().get_grad()).collect();
		(self.synapse.make_diff_node)(&self.inputs, &grads, self.synapse.ref_option())
    }

}
