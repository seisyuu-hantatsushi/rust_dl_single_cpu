/* -*- tab-width:4 -*- */
use num;

use crate::tensor::tensor_base::Tensor;

impl<T> Tensor<T>
where T:num::Num+Clone+Copy+std::fmt::Display {

	fn sum_subtensor(v:&[T],
					 src_shape:&[usize],
					 dst_shape:&[usize]) -> Tensor<T> {
		let t = if dst_shape.len() > 2 {
			let t = if dst_shape[0] == 1 {
				let stride = src_shape[1..].iter().fold(1,|prod, &e| prod*e);
				let strided_vec = v.chunks(stride);
				let init_accum:Vec<T> = vec![num::zero();stride];
				let sum_tensor_v = strided_vec.fold(init_accum,
													|accum, e| {
														accum.iter().zip(e.iter()).map(|(&a,&v)| a+v).collect::<Vec<T>>()
													});
				Self::sum_subtensor(&sum_tensor_v, &src_shape[1..], &dst_shape[1..])
			}
			else {
				let mut ts:Vec<Tensor<T>> = vec!();
				let stride = src_shape[1..].iter().fold(1,|prod, &e| prod*e);
				let strided_vec = v.chunks(stride);
				for sv in strided_vec {
					let t = Self::sum_subtensor(sv, &src_shape[1..], &dst_shape[1..]);
					ts.push(t);
				}
				let mut v:Vec<T> = vec!();
				for t in ts.iter() {
					v.extend(t.buffer());
				}

				fn new_reshaper(shape:&[usize]) -> Vec<usize> {
					if shape.len() > 2 {
						let mut v:Vec<usize> = vec!();
						if shape[0] != 1 {
							v.push(shape[0]);
						}
						v.extend(&new_reshaper(&shape[1..]));
						v
					}
					else {
						if shape[0] == 1 && shape[1] == 1 {
							vec![1]
						}
						else if shape[0] != 1 && shape[1] != 1 {
							shape.to_vec()
						}
						else {
							vec![shape[0]*shape[1]]
						}
					}
				}
				let new_shape = new_reshaper(&dst_shape);
				Tensor::from_vector(new_shape, v)
			};
			t
		}
		else {
			if dst_shape[0] == 1 && dst_shape[1] == 1 {
				let s = v.into_iter().fold(num::zero(),|accum, e| accum+*e);
				Tensor::<T>::from_array(&[1,1], &[s])
			}
			else if dst_shape[0] == 1 {
				let strided_vec = v.chunks(dst_shape[1]);
				let mut accum:Vec<T> = vec![num::zero();dst_shape[1]];
				for v in strided_vec {
					accum = accum.iter().zip(v.iter()).map(|(&a,&v)| a+v).collect::<Vec<T>>();
				}
				Tensor::from_vector(vec![1,dst_shape[1]],accum)
			}
			else if dst_shape[1] == 1 {
				let strided_vec = v.chunks(src_shape[1]);
				let mut sum_v:Vec<T> = vec!();
				for v in strided_vec {
					let s = v.iter().fold(num::zero(),|accum,&e| accum + e);
					sum_v.push(s);
				}
				Tensor::from_vector(vec![dst_shape[0],1],sum_v)
			}
			else {
				Tensor::<T>::from_array(dst_shape, v)
			}
		};
		return t;
	}

	pub fn sum(&self, shape:&[usize]) -> Tensor<T> {
		// まずは,形のチェック
		assert_eq!(shape.len(),self.shape().len());
		for (o,s) in self.shape().iter().zip(shape.iter()) {
			if !(o == s || *s == 1) {
				panic!("invalid shape {} {}",o,s);
			}
		}
		Self::sum_subtensor(self.buffer(), self.shape(), shape)
    }

	pub fn sum_axis(&self, axis:usize) -> Tensor<T> {
		if self.shape().len() == 0 {
			panic!("invalid source shape.");
		}
		else if self.shape().len() == 1 {
			if axis != 0 {
				panic!("invalid source shape.");
			}
			else {
				Tensor::<T>::from_vector(vec!(), vec![self.buffer().iter().fold(num::zero(),|s,&e| s+e)])
			}
		}
		else {
			let dst_shape = {
				let mut v = self.shape().to_vec();
				v[axis] = 1;
				v
			};
			self.sum(&dst_shape)
		}
	}
}
