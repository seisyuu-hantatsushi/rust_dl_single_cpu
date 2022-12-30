/* -*- tab-width:4 -*- */

use std::cell::RefCell;
use std::rc::Rc;
use linear_transform::tensor::Tensor;
use crate::synapse::{Synapse,SynapseOption,SynapseNode,NNSynapseNode};
use crate::neuron::{NeuronPrimType,NNNeuron,nn_neuron_new,nn_neuron_constant};

impl<T> SynapseNode<T>
where T:NeuronPrimType<T> {

    fn affine_forward(inputs: Vec<&Tensor<T>>, _opt: &Option<SynapseOption<T>>)
					  -> Vec<Tensor<T>> {
		let x_shape = inputs[0].shape();
		let w_shape = inputs[1].shape();
		assert_eq!(x_shape.len(),2);
		assert_eq!(w_shape.len(),2);
		assert_eq!(x_shape[1], w_shape[0]);
		let result_shape = vec![x_shape[0],w_shape[1]];
		let b = inputs[2].broadcast(&result_shape);
		let y = Tensor::<T>::affine(inputs[0], inputs[1], &b);
		vec![y]
    }

    fn affine_backward(inputs: &Vec<NNNeuron<T>>,
					   grads: &Vec<NNNeuron<T>>,
					   _opt: &Option<SynapseOption<T>>)
					   -> (Vec<NNSynapseNode<T>>,Vec<NNNeuron<T>>) {
		let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
		let mut outputs:Vec<NNNeuron<T>> = Vec::new();
		let (x,w,b) = (Rc::clone(&inputs[0]), Rc::clone(&inputs[1]), Rc::clone(&inputs[2]));

		if x.borrow().is_constant() &&
			w.borrow().is_constant() &&
			b.borrow().is_constant() {
				return (sns, outputs)
			}

		outputs.push(Rc::clone(&grads[0]));
		{
			let mut borrowed_b = b.borrow_mut();
			if !borrowed_b.is_constant() {
				let grad = if grads[0].borrow().ref_signal().shape() != borrowed_b.shape() {
					let (sn, output) = Self::sum_to(Rc::clone(&grads[0]), borrowed_b.shape());
					sns.push(sn);
					outputs.push(Rc::clone(&output));
					output
				}
				else {
					Rc::clone(&grads[0])
				};

				if let Some(ref g) = borrowed_b.ref_grad() {
					outputs.push(Rc::clone(g));
					let (sn, output) = Self::add(Rc::clone(&g), grad);
					output.borrow_mut().rename(&format!("{}+",g.borrow().name()));
					borrowed_b.set_grad(Rc::clone(&output));
					sns.push(sn);
					outputs.push(output)
				}
				else {
					grad.borrow_mut().rename(&format!("({})'",borrowed_b.name()));
					borrowed_b.set_grad(grad)
				}
			}
		}

		{
			let mut borrowed_w = w.borrow_mut();
			if !borrowed_w.is_constant()
			{
				let grad = if x.borrow().is_constant() && grads[0].borrow().is_constant()
				{
					let borrowed_grad = grads[0].borrow();
					let gt = borrowed_grad.ref_signal();
					let xt = x.borrow().ref_signal().transpose();
					let label = x.borrow().name().to_string() + "*" + borrowed_grad.name();
					nn_neuron_constant(&label, Tensor::<T>::matrix_product(&xt,gt))
				}
				else {
					outputs.push(Rc::clone(&x));
					let (sn,xt) = Self::transpose(Rc::clone(&x));
					sns.push(sn);
					outputs.push(Rc::clone(&xt));
					let (sn,output) = Self::matrix_product(xt,Rc::clone(&grads[0]));
					sns.push(sn);
					output
				};

				outputs.push(Rc::clone(&grad));
				if let Some(ref g) = borrowed_w.ref_grad() {
					outputs.push(Rc::clone(g));
					let (sn, output) = Self::add(Rc::clone(&g), grad);
					output.borrow_mut().rename(&format!("{}+",g.borrow().name()));
					borrowed_w.set_grad(Rc::clone(&output));
					sns.push(sn);
					outputs.push(output)
				}
				else {
					grad.borrow_mut().rename(&format!("({})'",borrowed_w.name()));
					borrowed_w.set_grad(grad);
				}
			}
		}

		{
			let mut borrowed_x = x.borrow_mut();
			if !borrowed_x.is_constant() {
				let borrowed_grad = grads[0].borrow();
				let grad = if w.borrow().is_constant() && grads[0].borrow().is_constant() {
					let gt = borrowed_grad.ref_signal();
					let wt = w.borrow().ref_signal().transpose();
					let label = borrowed_grad.name().to_string() + "*" + x.borrow().name();
					nn_neuron_constant(&label, Tensor::<T>::matrix_product(gt,&wt))
				}
				else {
					outputs.push(Rc::clone(&w));
					let (sn,wt) = Self::transpose(Rc::clone(&w));
					sns.push(sn);
					outputs.push(Rc::clone(&wt));
					let (sn,output) = Self::matrix_product(Rc::clone(&grads[0]),wt);
					sns.push(sn);
					output
				};
				outputs.push(Rc::clone(&grad));
				if let Some(ref g) = borrowed_x.ref_grad() {
					outputs.push(Rc::clone(g));
					let (sn, output) = Self::add(Rc::clone(&g), grad);
					output.borrow_mut().rename(&format!("{}+",borrowed_x.name()));
					borrowed_x.set_grad(Rc::clone(&output));
					sns.push(sn);
					outputs.push(output)
				}
				else {
					grad.borrow_mut().rename(&format!("({})'",borrowed_x.name()));
					borrowed_x.set_grad(grad);
				}
			}
		}
		(sns,outputs)
	}

    pub fn affine(x:NNNeuron<T>, w:NNNeuron<T>, b:NNNeuron<T>)
				  -> (NNSynapseNode<T>, NNNeuron<T>) {
		let label = "affine";
		let output = nn_neuron_new::<T>(&label, Tensor::<T>::zero(&[1,1]));
		let s = Synapse::<T>::new(Self::affine_forward, Self::affine_backward);
		let sn = SynapseNode::<T>::new(&label, vec![Rc::clone(&x), Rc::clone(&w), Rc::clone(&b)], vec![Rc::clone(&output)], s);
		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn,output)
	}

}
