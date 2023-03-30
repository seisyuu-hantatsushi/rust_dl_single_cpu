/* -*- tab-width:4 -*- */
use num;

use crate::tensor::tensor_base::Tensor;

#[derive(Copy, Clone)]
enum Operator {
	MAX, MIN
}

impl<T> Tensor<T>
where T:num::Num+num::FromPrimitive+Clone+Copy+std::cmp::PartialOrd {

	fn compare_tuple(op: Operator, p:(&T,&T)) -> T {
		let (&m,&e) = p;
		match op {
			Operator::MAX => if m < e { e } else { m }
			Operator::MIN => if m > e { e } else { m }
		}
	}

	fn compare_values(op: Operator, a:&T, b:&T) -> T {
		match op {
			Operator::MAX => if a < b { *b } else { *a }
			Operator::MIN => if a > b { *b } else { *a }
		}
	}

	fn max_min_subtensor(v:&[T],src_shape:&[usize],dst_shape:&[usize], op:Operator) -> Tensor<T> {
		if dst_shape.len() > 2 {
			if dst_shape[0] == 1 {
				let stride = src_shape[1..].iter().fold(1,|p,&e| p*e);
				let mut strided_vecs = v.chunks(stride);
				let max_tensor_v = if let Some(first_v) = strided_vecs.nth(0) {
					strided_vecs.fold(first_v.to_vec(),
									  |max_v, v| {
										  let compare_tuple_max = |p| Self::compare_tuple(op, p);
										  max_v.iter().zip(v.iter()).map(compare_tuple_max).collect::<Vec<T>>()
									  })
				}
				else {
					panic!("no element in Tensor")
				};
				Self::max_min_subtensor(&max_tensor_v, &src_shape[1..], &dst_shape[1..],op)
			}
			else {
				let mut ts:Vec<Tensor<T>> = vec!();
				let stride = src_shape[1..].iter().fold(1,|prod, &e| prod*e);
				let strided_vec = v.chunks(stride);
				for sv in strided_vec {
					let t = Self::max_min_subtensor(sv, &src_shape[1..], &dst_shape[1..],op);
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
			}
		}
		else {
			if dst_shape[0] == 1 && dst_shape[1] == 1 {
				let s = v.into_iter().fold(num::zero::<T>(),|accum, e| accum+*e);
				panic!("need to impl");
				Tensor::<T>::from_array(&[1,1], &[s])
			}
			else if dst_shape[0] == 1 {
				let mut strided_vecs = v.chunks(dst_shape[1]);
				let max_tensor_v = if let Some(first_v) = strided_vecs.nth(0) {
					strided_vecs.fold(first_v.to_vec(),
									  |mv, v| {
										  mv.iter().zip(v.iter())
											  .map(|p| Self::compare_tuple(op, p)).collect::<Vec<T>>() })
				}
				else {
					panic!("no element in Tensor")
				};
				Tensor::from_vector(vec![1,dst_shape[1]],max_tensor_v)
			}
			else if dst_shape[1] == 1 {
				let strided_vecs = v.chunks(src_shape[1]);
				let max_tensor_v =
					strided_vecs.map(|v| {
						let compare_value_max = |a:T,b:T| Self::compare_values(op, &a, &b);
						if let Some(m) = v.to_vec().into_iter().reduce(compare_value_max) {
							m
						}
						else {
							panic!("no element in Tensor")
						}
					}).collect::<Vec<T>>();
				Tensor::from_vector(vec![dst_shape[0],1],max_tensor_v)
			}
			else {
				Tensor::<T>::from_array(dst_shape, v)
			}
		}
	}

	fn max_min_in_axis(&self, axis:usize, op:Operator) -> Tensor<T> {
		if self.shape().len() == 0 {
			self.clone()
		}
		else if self.shape().len() == 1 {
			let compare_value_max = |a:T,b:T| Self::compare_values(op, &a, &b);
			let max = if let Some(m) = self.buffer().to_vec().into_iter().reduce(compare_value_max) {
				m
			}
			else {
				panic!("invalid source shape.");
			};
			Tensor::<T>::from_vector(vec![1], vec![max])
		}
		else {
			let dst_shape = {
				let mut v = self.shape().to_vec();
				v[axis] = 1;
				v
			};
			Self::max_min_subtensor(self.buffer(), self.shape(), &dst_shape, op)
		}
	}

	pub fn max_in_axis(&self, axis:usize) -> Tensor<T> {
		self.max_min_in_axis(axis, Operator::MAX)
	}

	pub fn min_in_axis(&self, axis:usize) -> Tensor<T> {
		self.max_min_in_axis(axis, Operator::MIN)
    }

	fn max_min_index_subtensor(v:&[T], src_shape:&[usize], axis:usize, op:Operator) -> Tensor<T> {
		if axis == 0 {
			let compare = |a:&T, b:&T| -> bool {
				match op {
					Operator::MAX => a < b,
					Operator::MIN => a > b
				}
			};
			let element_size = src_shape[1..].iter().fold(1,|p,&s| p*s);
			let mut elements:Vec<(usize,T)> = Vec::with_capacity(element_size);
			let chunks = v.chunks(element_size);
			for (i,chunk) in chunks.enumerate() {
				if i == 0 {
					for &v in chunk {
						elements.push((0,v))
					}
				}
				else {
					for (j,&v) in chunk.iter().enumerate() {
						if compare(&(elements[j].1), &v) {
							elements[j] = (i,v)
						}
					}
				}
			}
			let result_v:Vec<T> = elements.iter().map(|(i,_)| num::FromPrimitive::from_usize(*i).unwrap() ).collect();
			return Tensor::from_vector(src_shape[1..].to_vec(),result_v);
		}
		else {
			let mut tv:Vec<Tensor<T>> = vec!();
			let chunk_size = src_shape[1..].iter().fold(1,|p,&s| p*s);
			for subtensor in v.chunks(chunk_size){
				tv.push(Self::max_min_index_subtensor(subtensor, &src_shape[1..], axis-1, op));
			}
			return Tensor::bind(tv);
		}
		Tensor::zero(&[1,1])
	}

	fn arg_max_min(&self, axis:usize, op:Operator) -> Tensor<T> {
		if self.shape().len() == 0 {
			self.clone()
		}
		else if self.shape().len() == 1 {
			let mut current_pos:usize = 0;
			let mut current_val = self.buffer()[0];
			let compare = |a:&T, b:&T| -> bool {
					match op {
						Operator::MAX => a < b,
						Operator::MIN => a > b
					}
				};
			for (i,val) in self.buffer().iter().enumerate() {
				if compare(&current_val, val) {
					current_pos = i;
					current_val = *val;
				}
			}
			Tensor::<T>::from_vector(vec![1], vec![num::FromPrimitive::from_usize(current_pos).unwrap()])
		}
		else {
			Self::max_min_index_subtensor(self.buffer(), self.shape(), axis, op)
		}
	}

	pub fn argmax(&self, axis:usize) -> Tensor<T> {
		assert!(axis < self.shape().len());
		self.arg_max_min(axis, Operator::MAX)
	}
}
