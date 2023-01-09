/* -*- tab-width:4 -*- */

use std::cell::RefCell;
use std::rc::Rc;

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Normal,Distribution};
use rand_pcg::Pcg64;

use linear_transform::tensor::Tensor;
use crate::neural_network::NeuralNetwork;
use crate::neuron::{NeuronPrimType,NNNeuron};

pub struct Linear<T>
where T: NeuronPrimType<T> {
	name: String,
	enable_bias: bool,
	out_size: usize,
	w: Option<NNNeuron<T>>,
	b: Option<NNNeuron<T>>,
	output: Option<NNNeuron<T>>
}

pub type NNLinear<T> = RefCell<Linear<T>>;

pub enum NNLayer<T>
where T: NeuronPrimType<T> {
    Linear(NNLinear<T>)
}

impl<T> NeuralNetwork<T>
where T:NeuronPrimType<T> {

	pub fn create_linear_layer(&mut self,
							   name: &str,
							   out_size:usize,
							   enable_bias:bool) -> NNLayer<T> {
		NNLayer::Linear(
			RefCell::new(
				Linear {
					name: name.to_string(),
					enable_bias: enable_bias,
					out_size: out_size,
					w: None,
					b: None,
					output: None
				}
			)
		)
	}

	pub fn layer_set_inputs(&mut self, layer:&mut NNLayer<T>, inputs:Vec<NNNeuron<T>>) -> Vec<NNNeuron<T>> {
		match layer {
			NNLayer::<T>::Linear(l) => {
				let mut ns = vec!();
				let mut borrowed_l = l.borrow_mut();
				if let None = borrowed_l.w {
					let mut rng = Pcg64::from_entropy();
					let normal_dist = Normal::new(0.0,1.0).unwrap_or_else(|e| panic!("{} {}:{}", e.to_string(), file!(), line!()));
					let in_size = inputs[0].borrow().shape()[1];
					let w_shape = vec![in_size, borrowed_l.out_size];
					let ws = (0..(w_shape[0]*w_shape[1])).map(|_| {
						let w:T = num::FromPrimitive::from_f64(normal_dist
															   .sample(&mut rng))
							.unwrap_or_else(||
											panic!("failed to get normal dist value"));
						let l:T = num::FromPrimitive::from_usize(in_size)
							.unwrap_or_else(||
											panic!("failed to get length"));
						let rate = (num::one::<T>()/l).sqrt();
						w * rate
					}).collect();
					let label = borrowed_l.name.clone() + "_w";
					let wn = self.create_neuron(&label, Tensor::<T>::from_vector(w_shape,ws));
					borrowed_l.w = Some(Rc::clone(&wn));
					let output = if borrowed_l.enable_bias {
						let label = borrowed_l.name.clone() + "_b";
						let b = self.create_neuron(&label,Tensor::<T>::zero(&[1,1]));
						let output = self.affine(Rc::clone(&inputs[0]), wn, Some(Rc::clone(&b)));
						borrowed_l.b = Some(b);
						output
					}
					else {
						self.affine(Rc::clone(&inputs[0]), wn, None)
					};
					borrowed_l.output = Some(Rc::clone(&output));
					ns.push(output);
				}
				else {
					panic!("already set input variable");
				}
				ns
			},
			_ => vec!()
		}
	}
}

impl<T> NNLayer<T>
where T:NeuronPrimType<T> {
	pub fn get_params(&self) -> Vec<NNNeuron<T>> {
		match self {
			NNLayer::<T>::Linear(l) => {
				let mut ns:Vec<NNNeuron<T>> = vec!();
				let borrowed_l = l.borrow();
				if let Some(ref w) = borrowed_l.w {
					ns.push(Rc::clone(w));
				};
				if let Some(ref b) = borrowed_l.b {
					ns.push(Rc::clone(b));
				}
				ns
			}
		}
	}
}
