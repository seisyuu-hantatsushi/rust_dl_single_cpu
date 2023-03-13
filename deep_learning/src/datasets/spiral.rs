/* -*- tab-width:4 -*- */

use std::fmt;
use num::FromPrimitive;
use linear_transform::tensor::Tensor;
use crate::neuron::NeuronPrimType;

use rand::prelude::SliceRandom;
use rand::SeedableRng;
use rand_distr::{Normal,Distribution,StandardNormal};
use rand_pcg::Pcg64;

pub fn get_2d_dataset<T>(num_of_class:usize, num_of_data:usize) -> (Tensor::<T>, Tensor::<T>) where T:NeuronPrimType<T>, StandardNormal: Distribution<T>
{
	let mut rng = Pcg64::from_entropy();
	let mut data:Vec<T>    = vec!();
	let mut teacher:Vec<T> = vec!();
	let normal_dist = Normal::<T>::new(num::zero(), num::one()).unwrap_or_else(|e| panic!("{} {}:{}", e.to_string(), file!(), line!()));
	let casted_n:T = num::FromPrimitive::from_usize(num_of_data).unwrap_or_else(||
																				panic!("{}:{}", file!(), line!()));
	let casted_nc:T = num::FromPrimitive::from_usize(num_of_class).unwrap_or_else(||
																				panic!("{}:{}", file!(), line!()));
	for j in 0..num_of_class {
		let casted_j:T = num::FromPrimitive::from_usize(j).unwrap_or_else(||
																		  panic!("{}:{}", file!(), line!()));
		for i in 0..num_of_data {
			let casted_i:T = num::FromPrimitive::from_usize(i).unwrap_or_else(||
																			  panic!("{}:{}", file!(), line!()));
			let rate   = casted_i/casted_n;
			let radius = rate;
			let theta = casted_j*(casted_nc+num::one()) + (casted_nc+num::one())*rate +
				normal_dist.sample(&mut rng) * num::FromPrimitive::from_f64(0.2).unwrap();
			let x = radius * theta.sin();
			let y = radius * theta.cos();
			data.push(x);
			data.push(y);
			teacher.push(casted_j);
		}
	}
	let perm_table = {
		let mut v = (0..(num_of_data*num_of_class)).collect::<Vec<usize>>();
		v.shuffle(&mut rng);
		v
	};
	let mut shuffled_data:Vec<T>    = vec!();
	let mut shuffled_teacher:Vec<T> = vec!();

	for &i in perm_table.iter() {
		shuffled_data.push(data[2*i]);
		shuffled_data.push(data[2*i+1]);
		shuffled_teacher.push(teacher[i]);
	}

	(Tensor::<T>::from_vector(vec![num_of_data*num_of_class,2],shuffled_data),
	 Tensor::<T>::from_vector(vec![1,num_of_data*num_of_class],shuffled_teacher))
}
