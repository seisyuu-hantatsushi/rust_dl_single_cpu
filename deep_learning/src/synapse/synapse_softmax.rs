/* -*- tab-width:4 -*- */

use std::fmt;
use std::cell::RefCell;
use std::rc::Rc;
use linear_transform::tensor::Tensor;
use crate::synapse::{Synapse,SynapseOption,SynapseNode,NNSynapseNode};
use crate::neuron::{NNNeuron, nn_neuron_new};

impl<T> SynapseNode<T>
where T:num::Float + num::FromPrimitive + num::pow::Pow<T, Output = T> + Clone + fmt::Display {

    fn softmax_forward(inputs: Vec<&Tensor<T>>, opt:&Option<SynapseOption<T>>) -> Vec<Tensor<T>> {
		let axis = if let Some(o) = opt {
			if let SynapseOption::Softmax(axis) = o {
				axis
			}
			else {
				panic!("Invalid Option")
			}
		}
		else {
			panic!("Invalid Option")
		};

		let src_shape = inputs[0].shape();
		let max = inputs[0].max_in_axis(*axis).broadcast(src_shape);
		//println!("src {}", inputs[0]);
		//println!("max {}", max);
		let y = (inputs[0]-max).exp();
		//println!("y {}", y);
		let sum_y = y.sum_axis(*axis).broadcast(src_shape);
		//println!("sum_y {}", sum_y);
		vec![Tensor::<T>::hadamard_division(&y, &sum_y)]
    }

    fn softmax_backword(inputs: &Vec<NNNeuron<T>>,
						grads: &Vec<NNNeuron<T>>,
						opt:&Option<SynapseOption<T>>) -> (Vec<NNSynapseNode<T>>, Vec<NNNeuron<T>>) {
		let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
		let mut outputs:Vec<NNNeuron<T>> = Vec::new();
		let &axis = if let Some(o) = opt {
			if let SynapseOption::Softmax(axis) = o {
				axis
			}
			else {
				panic!("Invalid Option")
			}
		}
		else {
			panic!("Invalid Option")
		};

		let (sn,y) = Self::softmax(Rc::clone(&inputs[0]), axis);
		sns.push(sn);
		outputs.push(Rc::clone(&y));

		let (sn,gx) = Self::hadamard_product(Rc::clone(&y),Rc::clone(&grads[0]));
		sns.push(sn);
		outputs.push(Rc::clone(&gx));

		let (sn,sumdx) = Self::sum_in_axis(Rc::clone(&gx), axis);
		sns.push(sn);
		outputs.push(Rc::clone(&sumdx));

		let dst_shape = sumdx.borrow().shape().to_vec();
		let (sn,dx) = Self::broadcast_to(sumdx, &dst_shape);
		sns.push(sn);
		outputs.push(Rc::clone(&dx));

		let (sn,gz) = Self::hadamard_product(y, Rc::clone(&dx));
		sns.push(sn);
		outputs.push(Rc::clone(&gz));

		let (sn,gx) = Self::sub(gx, gz);
		sns.push(sn);
		outputs.push(Rc::clone(&gx));

		let mut n = inputs[0].borrow_mut();
		if let Some(ref g) = n.ref_grad() {
			let (sn, output) = Self::add(Rc::clone(&g), Rc::clone(&gx));
			sns.push(sn);
			output.borrow_mut().rename(&format!("{}+", g.borrow().name()));
			outputs.push(Rc::clone(&output));
			n.set_grad(output)
		}
		else {
			gx.borrow_mut().rename(&format!("({})'", n.name()));
			n.set_grad(gx)
		}

		(sns,outputs)
    }

    pub fn softmax(x:NNNeuron<T>, axis:usize) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "softmax";
		let output = nn_neuron_new::<T>(&label, Tensor::<T>::zero(&[1,1]));
		let s = Synapse::<T>::new_with_option(Self::softmax_forward,
											  Self::softmax_backword,
											  SynapseOption::Softmax(axis));
		let sn = SynapseNode::<T>::new(&label,
									   vec![Rc::clone(&x)],
									   vec![Rc::clone(&output)],
									   s);
		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn,output)
    }
}
