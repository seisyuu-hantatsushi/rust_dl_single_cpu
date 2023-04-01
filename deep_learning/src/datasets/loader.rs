/* -*- tab-width:4 -*- */

use linear_transform::tensor::Tensor;
use crate::datasets::*;
use crate::neuron::NeuronPrimType;

use rand_pcg::Pcg64;
use rand::SeedableRng;
use rand::prelude::SliceRandom;

pub trait LoaderSource<T>
where T:NeuronPrimType<T> {
    fn load_batch(&mut self, index:usize, batch_size:usize, permute_table:&Option<Vec<usize>>)
				  -> Option<(Tensor<T>,Tensor<T>)>;
    fn get_num_of_image(&self) -> usize;
	fn get_data_shape(&self) -> Vec<usize>;
}

pub struct Loader<T> where T: NeuronPrimType<T> {
	rng: Option<Pcg64>,
	data_inst  : Box<dyn LoaderSource<T>>,
	batch_size : usize,
	permute_table: Option<Vec<usize>>
}

pub struct Batchs<'a, T> where T: NeuronPrimType<T> {
    loader: &'a mut Loader<T>,
    current: usize
}

impl<'a, T> Iterator for Batchs<'a, T> where T: NeuronPrimType<T> {
	type Item = (Tensor<T>,Tensor<T>);
	fn next(&mut self) -> Option<Self::Item> {
		let current = self.current;
		self.current += 1;
		self.loader.data_inst.load_batch(current, self.loader.batch_size, &self.loader.permute_table)
	}
}

impl<T> Loader<T>
where T: NeuronPrimType<T> {
    pub fn new(inst:Box<dyn LoaderSource<T>>, batch_size:usize, shuffle:bool) -> Loader<T> {

		let mut rng = if shuffle
		{
			Some(Pcg64::from_entropy())
		}
		else
		{
			None
		};

		Loader {
			rng : rng,
			data_inst: inst,
			batch_size: batch_size,
			permute_table: None
		}

    }

    pub fn get_batchs(&mut self) -> Batchs<T> {
		self.permute_table = if let Some(ref mut rng) = self.rng
		{
			let mut permute_table = (0..self.data_inst.get_num_of_image()).collect::<Vec<usize>>();
			permute_table.shuffle(rng);
			Some(permute_table)
		}
		else {
			None
		};
		Batchs {
			loader: self,
			current: 0
		}
    }

	pub fn get_data_shape(&mut self) -> Vec<usize> {
		self.data_inst.get_data_shape()
	}
}
