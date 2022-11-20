/* -*- tab-width:4 -*- */
use std::fmt;
use std::cell::{Ref,RefCell};
use std::rc::Rc;
use linear_transform::tensor::Tensor;

use crate::neuron::{NNNeuron,Neuron,nn_neuron_new,nn_neuron_constant};

pub type MakeDiffNode<T> = fn (inputs: &Vec<NNNeuron<T>>, grads: &Vec<NNNeuron<T>>) -> (Vec<NNSynapseNode<T>>,Vec<NNNeuron<T>>);

pub struct Synapse<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone {
	forward: fn (inputs: Vec<&Tensor<T>>) -> Vec<Tensor<T>>,
	make_diff_node: MakeDiffNode<T>
}

impl<T> Synapse<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone {
	pub fn new(forward: fn (inputs: Vec<&Tensor<T>>) -> Vec<Tensor<T>>,
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
		//let label = "-".to_string()+x.borrow().name();
		let label = "neg";
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
										  }
									  });
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
	}

	pub fn add(x:NNNeuron<T>, y:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "add";
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
											  let mut outputs:Vec<NNNeuron<T>> = Vec::new();
											  outputs.push(Rc::clone(&grads[0]));
											  for ni in inputs.iter() {
												  let mut n = ni.borrow_mut();
												  if !n.is_constant() {
													  if let Some(ref g) = n.ref_grad() {
														  outputs.push(Rc::clone(&g));
														  let (sn, output) = Self::add(Rc::clone(&g), Rc::clone(&grads[0]));
														  n.set_grad(Rc::clone(&output));
														  sns.push(Rc::clone(&sn));
														  outputs.push(output);
													  }
													  else {
														  n.set_grad(Rc::clone(&grads[0]))
													  }
												  }
											  }
											  (sns,outputs)
										  }
									  });
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
	}

	pub fn sub(x:NNNeuron<T>, y:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		//let label = "(".to_string()+x.borrow().name() + "-" + y.borrow().name() +")";
		let label = "sub";
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
											  let mut outputs:Vec<NNNeuron<T>> = Vec::new();
											  let mut input_grad = false;
											  let mut n = inputs[0].borrow_mut();
											  if !n.is_constant() {
												  if let Some(ref g) = n.ref_grad() {
													  outputs.push(Rc::clone(&g));
													  let (sn, output) = Self::add(Rc::clone(&g), Rc::clone(&grads[0]));
													  n.set_grad(Rc::clone(&output));
													  sns.push(sn);
													  outputs.push(output);
													  input_grad = true;
												  }
												  else {
													  n.set_grad(Rc::clone(&grads[0]));
												  }
											  }

											  let mut n = inputs[1].borrow_mut();
											  if !n.is_constant() {
												  let (sn,output) = Self::neg(Rc::clone(&grads[0]));
												  sns.push(sn);
												  input_grad = true;
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
											  if input_grad {
												  outputs.push(Rc::clone(&grads[0]));
											  }
											  (sns,outputs)
										  }
									  });
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
	}

	pub fn exp(x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
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
										  }
									  });
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
	}

	pub fn pow_rank0(a:NNNeuron<T>, x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		assert_eq!(x.borrow().shape(),&[1,1]);
		assert_eq!(a.borrow().shape(),&[1,1]);
		let label = "pow";
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
											  let mut outputs:Vec<NNNeuron<T>> = Vec::new();
											  if !inputs[0].borrow().is_constant() {
												  let dec_index = if inputs[1].borrow().is_constant() {
													  let t = inputs[1].borrow().ref_signal() - Tensor::<T>::one(&[1,1]);
													  let label = inputs[1].borrow().name().to_string() + "-1.0";
													  Rc::new(RefCell::new(Neuron::<T>::constant(&label, t)))
												  }
												  else {
													  let one = Rc::new(RefCell::new(Neuron::<T>::constant("1.0", Tensor::<T>::one(&[1,1]))));
													  outputs.push(Rc::clone(&one));
													  let (dec_index_sn, dec_index) = Self::sub(Rc::clone(&inputs[1]),one);
													  sns.push(dec_index_sn);
													  dec_index
												  };
												  outputs.push(Rc::clone(&dec_index));
												  let (l_sn,l_output) = Self::pow_rank0(Rc::clone(&inputs[0]), dec_index);
												  outputs.push(Rc::clone(&inputs[0]));
												  sns.push(l_sn);
												  outputs.push(Rc::clone(&l_output));
												  let (l_sn,l_output) = Self::mul_rank0(Rc::clone(&inputs[1]), l_output);
												  outputs.push(Rc::clone(&inputs[1]));
												  sns.push(l_sn);
												  outputs.push(Rc::clone(&l_output));
												  let (l_sn,l_output) = Self::mul_rank0(Rc::clone(&grads[0]),  l_output);
												  sns.push(l_sn);
												  outputs.push(Rc::clone(&l_output));
												  outputs.push(Rc::clone(&grads[0]));
												  {
													  let mut n = inputs[0].borrow_mut();
													  if let Some(ref g) = n.ref_grad() {
														  outputs.push(Rc::clone(&g));
														  let (sn, output) = Self::add(Rc::clone(&g), l_output);
														  n.set_grad(Rc::clone(&output));
														  sns.push(sn);
														  outputs.push(output);
													  }
													  else {
														  n.set_grad(l_output);
													  }
												  }
											  }
											  if !inputs[1].borrow().is_constant() {
												  let (r_sn,r_output) = Self::pow_rank0(Rc::clone(&inputs[0]),Rc::clone(&inputs[1]));
												  sns.push(r_sn);
												  outputs.push(Rc::clone(&r_output));
												  let (baselog_sn, baselog) = Self::ln_rank0(Rc::clone(&inputs[0]));
												  sns.push(baselog_sn);
												  outputs.push(Rc::clone(&baselog));
												  let (r_sn,r_output) = Self::mul_rank0(baselog,r_output);
												  sns.push(r_sn);
												  outputs.push(Rc::clone(&r_output));
												  {
													  let mut n = inputs[1].borrow_mut();
													  if let Some(ref g) = n.ref_grad() {
														  outputs.push(Rc::clone(&g));
														  let (sn, output) = Self::add(Rc::clone(&g), r_output);
														  n.set_grad(Rc::clone(&output));
														  sns.push(sn);
														  outputs.push(output);
													  }
													  else {
														  n.set_grad(r_output);
													  }
												  }
											  }
											  (sns,outputs)
										  }
									  });
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
	}

	pub fn ln_rank0(x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "ln_rank0";
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
											  let mut outputs:Vec<NNNeuron<T>> = Vec::new();
											  // (ln x)' = 1/x
											  let one = nn_neuron_constant("1.0", Tensor::<T>::one(&[1,1]));
											  outputs.push(Rc::clone(&one));
											  let (inverse_sn, inverse) = Self::div_rank0(one, Rc::clone(&inputs[0]));
											  sns.push(inverse_sn);
											  outputs.push(Rc::clone(&inverse));
											  let (sn, output) = Self::mul_rank0(Rc::clone(&grads[0]), inverse);
											  sns.push(sn);
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
										  }
									  });
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
	}

	pub fn hadamard_product(x:NNNeuron<T>,y:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>){
		assert_eq!(x.borrow().shape(),y.borrow().shape());
		let label = "hadamard";
		let output = nn_neuron_new::<T>(&label, Tensor::<T>::zero(&[1,1]));
		let s = SynapseNode::<T>::new(&label,
									  vec![Rc::clone(&x), Rc::clone(&y)],
									  vec![Rc::clone(&output)],
									  Synapse::<T>::new(
										  |inputs| {
											  vec![Tensor::<T>::hadamard_product(inputs[0], inputs[1])]
										  },
										  |inputs,grads| {
											  let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
											  let mut outputs:Vec<NNNeuron<T>> = Vec::new();

											  if inputs[0].borrow().is_constant() &&
												 inputs[1].borrow().is_constant() {
												  (sns,outputs)
											  }
											  else {
												  if !inputs[0].borrow().is_constant() {
													  let output = if grads[0].borrow().is_constant() && inputs[1].borrow().is_constant() {
														  let t = Tensor::<T>::hadamard_product(grads[0].borrow().ref_signal(),inputs[1].borrow().ref_signal());
														  let label = "(".to_string() + grads[0].borrow().name() + ") * (" + inputs[1].borrow().name() + ")";
														  nn_neuron_constant(&label, t)
													  }
													  else {
														  let (l_sn, l_output) = Self::hadamard_product(Rc::clone(&grads[0]), Rc::clone(&inputs[1]));
														  sns.push(l_sn);
														  outputs.push(Rc::clone(&inputs[1]));
														  outputs.push(Rc::clone(&grads[0]));
														  l_output
													  };
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
												  }
												  if !inputs[1].borrow().is_constant() {
													  let output = if grads[0].borrow().is_constant() && inputs[0].borrow().is_constant() {
														  let t = Tensor::<T>::hadamard_product(grads[0].borrow().ref_signal(),inputs[0].borrow().ref_signal());
														  let label = "(".to_string() + grads[0].borrow().name() + ") * (" + inputs[0].borrow().name() + ")";
														  nn_neuron_constant(&label, t)
													  }
													  else {
														  let (l_sn, l_output) = Self::hadamard_product(Rc::clone(&grads[0]), Rc::clone(&inputs[0]));
														  sns.push(l_sn);
														  outputs.push(Rc::clone(&inputs[0]));
														  outputs.push(Rc::clone(&grads[0]));
														  l_output
													  };
													  outputs.push(Rc::clone(&output));
													  {
														  let mut n = inputs[1].borrow_mut();
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
												  }
												  (sns,outputs)
											  }
										  }
									  ));
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
										  |inputs| {
											  vec![inputs[0].tanh()]
										  },
										  |inputs,grads| {
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
		let outputs = (self.synapse.forward)(inputs);
		self.outputs.iter().zip(outputs.into_iter()).map(|(n,t)| {n.borrow_mut().assign(t); Rc::clone(n)}).collect()
    }

    pub fn make_diff_node(&self) -> (Vec<NNSynapseNode<T>>, Vec<NNNeuron<T>>) {
		let grads =
			self.outputs.iter().map(|no| no.borrow_mut().get_grad()).collect();
		(self.synapse.make_diff_node)(&self.inputs, &grads)
    }

}
