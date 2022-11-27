use num;

use crate::tensor::tensor_base::{Tensor};

impl<T> Tensor<T>
where T:num::Num+Clone+Copy+std::fmt::Debug {
    pub fn reshape(&self, shape:&[usize]) -> Tensor<T> {
	assert_eq!(self.buffer().len(), shape.iter().fold(1,|prod,d| { prod * (*d) }));
	Tensor::from_vector(shape.to_vec(), self.buffer().to_vec())
    }

    fn sum_subtensor(v:&[T],
		     src_shape:&[usize],
		     dst_shape:&[usize]) -> Tensor<T> {
	println!("src_shape:{:?}, dst_shape:{:?}", src_shape, dst_shape);
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
		    println!("dst_shape:{:?}",shape);
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
		println!("new_shape {:?}", new_shape);
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
	let mut dim = 0;

	// まずは,形のチェック
	assert_eq!(shape.len(),self.shape().len());
	for (o,s) in self.shape().iter().zip(shape.iter()) {
	    if !(o == s || *s == 1) {
		panic!("invalid shape");
	    }
	}
	Self::sum_subtensor(self.buffer(), self.shape(), shape)
    }
}

#[test]
fn tensor_reshape_test () {
    {
	let shape:[usize;2] = [3,4];
	let m_init:[f32;12] = [ 11.0,12.0,13.0,14.0,
				21.0,22.0,23.0,24.0,
				31.0,32.0,33.0,34.0 ];
	let t0 = Tensor::<f32>::from_array(&shape, &m_init);
	let t1 = t0.reshape(&[4,3]);
	println!("reshape {}", t1);

	let t1 = t0.sum(&[1,1]);
	println!("sum [1,1] {} {}", t1,IntoIterator::into_iter(m_init).fold(0.0, |accum, e| accum+e));
	assert_eq!(t1, Tensor::<f32>::from_array(&[1,1],&[IntoIterator::into_iter(m_init).fold(0.0, |accum, e| accum+e)]));

	let t1 = t0.sum(&[1,4]);
	println!("sum [1,4] {}", t1);
	assert_eq!(t1, Tensor::<f32>::from_array(&[1,4],&[63.0,66.0,69.0,72.0]));

	let t1 = t0.sum(&[3,1]);
	println!("sum [3,1] {}", t1);
	assert_eq!(t1, Tensor::<f32>::from_array(&[3,1],&[50.0,90.0,130.0]));
    }
    {
	let m_init:[f32;36] = [ 111.0,112.0,113.0,114.0,
				121.0,122.0,123.0,124.0,
				131.0,132.0,133.0,134.0,
				211.0,212.0,213.0,214.0,
				221.0,222.0,223.0,224.0,
				231.0,232.0,233.0,234.0,
				311.0,312.0,313.0,314.0,
				321.0,322.0,323.0,324.0,
				331.0,332.0,333.0,334.0 ];
	let shape:[usize;3] = [3,3,4];
	let t0 = Tensor::<f32>::from_array(&shape, &m_init);
	let t1 = t0.sum(&[1,1,1]);
	println!("sum [1,1,1] {} {}", t1,IntoIterator::into_iter(m_init).fold(0.0, |accum, e| accum+e));
	assert_eq!(t1, Tensor::<f32>::from_array(&[1,1],&[IntoIterator::into_iter(m_init).fold(0.0, |accum, e| accum+e)]));

	let t1 = t0.sum(&[1,3,4]);
	println!("sum [1,3,4] {}", t1);
	assert_eq!(t1, Tensor::<f32>::from_array(&[3,4],
						 &[633.0,636.0,639.0,642.0,
						   663.0,666.0,669.0,672.0,
						   693.0,696.0,699.0,702.0]));

	let t1 = t0.sum(&[3,1,4]);
	println!("sum [3,1,4] {}", t1);

	let t1 = t0.sum(&[3,1,1]);
	println!("sum [3,1,1] {}", t1);

	let t1 = t0.sum(&[1,3,1]);
	println!("sum [1,3,1] {}", t1);

    }

    {
	let m_init:[f32;72] = [ 111.0,112.0,113.0,114.0,
				121.0,122.0,123.0,124.0,
				131.0,132.0,133.0,134.0,
				211.0,212.0,213.0,214.0,
				221.0,222.0,223.0,224.0,
				231.0,232.0,233.0,234.0,
				311.0,312.0,313.0,314.0,
				321.0,322.0,323.0,324.0,
				331.0,332.0,333.0,334.0,

				111.0,112.0,113.0,114.0,
				121.0,122.0,123.0,124.0,
				131.0,132.0,133.0,134.0,
				211.0,212.0,213.0,214.0,
				221.0,222.0,223.0,224.0,
				231.0,232.0,233.0,234.0,
				311.0,312.0,313.0,314.0,
				321.0,322.0,323.0,324.0,
				331.0,332.0,333.0,334.0 ];
	let shape:[usize;4] = [2,3,3,4];
	let t0 = Tensor::<f32>::from_array(&shape, &m_init);
	let t1 = t0.sum(&[1,1,1,1]);
	println!("sum [1,1,1,1] {} {}", t1,IntoIterator::into_iter(m_init).fold(0.0, |accum, e| accum+e));
	assert_eq!(t1, Tensor::<f32>::from_array(&[1,1],&[IntoIterator::into_iter(m_init).fold(0.0, |accum, e| accum+e)]));

	let t1 = t0.sum(&[1,3,3,4]);
	println!("sum [1,3,3,4] {}", t1);

	let t1 = t0.sum(&[2,3,3,1]);
	println!("sum [2,3,3,1] {}", t1);

    }
}
