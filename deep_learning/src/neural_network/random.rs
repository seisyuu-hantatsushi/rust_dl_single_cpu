/* -*- tab-width:4 -*- */

use std::cell::RefCell;
use std::rc::Rc;

use rand::SeedableRng;
use rand_distr::{Normal,Distribution,StandardNormal};

use linear_transform::tensor::Tensor;
use crate::neural_network::NeuralNetwork;
use crate::neuron::{NeuronPrimType,NNNeuron};

impl<T> NeuralNetwork<T>
where T:NeuronPrimType<T>, StandardNormal: Distribution<T> {
    pub fn create_normal_distribution_neuron(&mut self, label:&str, shape:&[usize], mean:T, dist:T) -> NNNeuron<T> {
	let element_size = shape.iter().fold(1,|p,&e| { p*e });
	let normal_dist = Normal::<T>::new(mean,dist).unwrap_or_else(|e| panic!("{} {}:{}", e.to_string(), file!(), line!()));
	let v:Vec<T> = Iterator::map(0..element_size, |_| { normal_dist.sample(&mut self.get_rng()) }).collect();
	let t = Tensor::<T>::from_vector(shape.to_vec(), v);
	self.create_neuron(label, t)
    }
}
