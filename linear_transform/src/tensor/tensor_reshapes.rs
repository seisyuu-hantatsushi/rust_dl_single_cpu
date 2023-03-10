/* -*- tab-width:4 -*- */
use num;

use crate::tensor::tensor_base::Tensor;

impl<T> Tensor<T>
where T:num::Num + Clone + Copy {

	pub fn reshape(&self, shape:&[usize]) -> Tensor<T> {
		assert_eq!(self.buffer().len(), shape.iter().fold(1,|prod,d| { prod * (*d) }));
		Tensor::from_vector(shape.to_vec(), self.buffer().to_vec())
	}

    fn broadcast_subtensor(v:&[T],
						   src_shape:&[usize],
						   dst_shape:&[usize]) -> Vec<T> {
		//println!("broadcast subtensor {:?} {:?}",src_shape,dst_shape);
		let v = if src_shape.len() > 1 {
			let mut broadcasted_v:Vec<T> = Vec::new();
			if src_shape[0] == 1 {
				for _ in 0..dst_shape[0] {
					broadcasted_v.extend(Self::broadcast_subtensor(v, &src_shape[1..], &dst_shape[1..]));
				}
			}
			else {
				let stride = src_shape[1..].iter().fold(1,|prod, &e| prod*e);
				for v in v.chunks(stride) {
					broadcasted_v.extend(Self::broadcast_subtensor(v, &src_shape[1..], &dst_shape[1..]));
				}
			}
			broadcasted_v
		}
		else {
			assert!(dst_shape[0] % src_shape[0] == 0);
			if dst_shape[0] != src_shape[0] {
				let mut broadcasted_v:Vec<T> = Vec::new();
				for _ in 0..dst_shape[0] {
					broadcasted_v.extend(v);
				}
				broadcasted_v
			}
			else {
				v.to_vec()
			}
		};
		v
    }

    pub fn broadcast(&self, shape:&[usize]) -> Tensor<T> {
		assert!(self.shape().len() <= shape.len());
		let (src_shape, dst_shape):(Vec<usize>,Vec<usize>) = if self.shape().len() == shape.len() {
			(self.shape().to_vec(), shape.to_vec())
		}
		else {
			let mut src_shape:Vec<usize> = vec![1;shape.len()-self.shape().len()];
			src_shape.extend(&self.shape().to_vec());
			(src_shape,shape.to_vec())
		};
		let v = Self::broadcast_subtensor(self.buffer(), &src_shape, &dst_shape);
		Tensor::from_vector(dst_shape.to_vec(), v)
    }

	pub fn ravel(&self) -> Tensor<T> {
		let shape = vec![1, self.num_of_elements()];
		Tensor::<T>::from_vector(shape,self.buffer().to_vec())
	}

	pub fn selector(&self, selector:&[usize]) -> Tensor<T> {
		let shape = {
			if self.shape().len() == 2 {
				let mut s = self.shape().to_vec();
				if s[0] == 1 {
					s[1] = selector.len();
				}
				else {
					s[0] = selector.len();
				}
				s
			}
			else {
				let mut s = self.shape().to_vec();
				s[0] = selector.len();
				s
			}
		};
		let mut v:Vec<T> = vec!();
		for &s in selector.iter() {
			let st = self.subtensor(s);
			v.extend_from_slice(st.buffer());
		}
		Tensor::<T>::from_vector(shape,v)
	}

}
