/* -*- tab-width:4 -*- */
use linear_transform::Tensor;

#[test]
fn tensor_test() {
    let shape:[usize;1] = [1];
    let t = Tensor::<f64>::zero(&shape);
    println!("{}", t);

    let shape:[usize;1] = [3];
    let t = Tensor::<f64>::zero(&shape);
    println!("{}", t);
    let m_init:[f32;12] = [ 11.0,12.0,13.0,14.0,
			    21.0,22.0,23.0,24.0,
			    31.0,32.0,33.0,34.0 ];
    let t = Tensor::<f32>::from_array(&[3,4],&m_init);
    println!("{}", t);
    let m_init:[f32;36] = [ 111.0,112.0,113.0,114.0,
			    121.0,122.0,123.0,124.0,
			    131.0,132.0,133.0,134.0,
			    211.0,212.0,213.0,214.0,
			    221.0,222.0,223.0,224.0,
			    231.0,232.0,233.0,234.0,
			    311.0,312.0,313.0,314.0,
			    321.0,322.0,323.0,324.0,
			    331.0,332.0,333.0,334.0 ];
    let t = Tensor::<f32>::from_array(&[3,3,4],&m_init);
    println!("{}", t);

    let m_init:[f32;72] = [ 1111.0,1112.0,1113.0,1114.0,
			    1121.0,1122.0,1123.0,1124.0,
			    1131.0,1132.0,1133.0,1134.0,
			    1211.0,1212.0,1213.0,1214.0,
			    1221.0,1222.0,1223.0,1224.0,
			    1231.0,1232.0,1233.0,1234.0,
			    1311.0,1312.0,1313.0,1314.0,
			    1321.0,1322.0,1323.0,1324.0,
			    1331.0,1332.0,1333.0,1334.0,
			    2111.0,2112.0,2113.0,2114.0,
			    2121.0,2122.0,2123.0,2124.0,
			    2131.0,2132.0,2133.0,2134.0,
			    2211.0,2212.0,2213.0,2214.0,
			    2221.0,2222.0,2223.0,2224.0,
			    2231.0,2232.0,2233.0,2234.0,
			    2311.0,2312.0,2313.0,2314.0,
			    2321.0,2322.0,2323.0,2324.0,
			    2331.0,2332.0,2333.0,2334.0 ];
    let t = Tensor::<f32>::from_array(&[2,3,3,4],&m_init);
    println!("{}", t);

    println!("{}", t[vec![1,2,2,3]]);

    let st = t.sub_tensor(1);
    println!("{}", st);

    let st = st.sub_tensor(0);
    println!("st 2");
    println!("{}", st);

    let mut t = Tensor::<f32>::from_array(&[2,3,3,4],&m_init);
    assert_eq!(t[vec![0,2,2,3]],1334.0);
    t.set(&[0,2,2,3], 0.0);
    assert_eq!(t[vec![0,2,2,3]],0.0);
    t[vec![0,2,2,3]] = 1335.0;
    assert_eq!(t[vec![0,2,2,3]],1335.0);
    assert_eq!(t[vec![0,2,2,2]],1333.0);

    let t = Tensor::<f64>::from_array(&[1,5], &[1.0, 0.0, 5.0, 1.0, 2.0]);
    assert_eq!(t.max_element_index(), (vec![0,2],5.0));
    let t = Tensor::<f64>::from_array(&[2, 5], &[1.0, 0.0, 5.0, 1.0, 2.0, 1.0, 0.6, 5.0, 6.0, 2.0]);
    assert_eq!(t.max_element_index(), (vec![1,3],6.0));

    let m_init:[f32;30] = [ 1.0, 0.0, 5.0, 1.0, 2.0,
			    1.0, 0.6, 5.0, 6.0, 2.0,

			    1.0, 0.0, 7.0, 1.0, 2.0,
			    1.0, 0.6, 5.0, 6.0, 2.0,

			    1.0, 0.0, 2.0, 1.0, 2.0,
			    1.0, 0.6, 5.0, 6.0, 2.0 ];
    let t = Tensor::<f32>::from_array(&[3,2,5],&m_init);
    assert_eq!(t.max_element_index(), (vec![1,0,2],7.0));
}

#[test]
fn tensor_index_to_position() {
    let t = Tensor::<f64>::from_array(&[2, 5, 4, 6], &[1.0f64;2*5*4*6]);

    let v = t.index_to_position(19);
    let i = v[0]*5*4*6 + v[1]*4*6 + v[2]*6 + v[3];
    assert_eq!(i, 19);

    let v = t.index_to_position(20);
    let i = v[0]*5*4*6 + v[1]*4*6 + v[2]*6 + v[3];
    assert_eq!(i, 20);

    let v = t.index_to_position(0);
    let i = v[0]*5*4*6 + v[1]*4*6 + v[2]*6 + v[3];
    assert_eq!(i, 0);

    let v = t.index_to_position(2*5*4*6/2+26);
    let i = v[0]*5*4*6 + v[1]*4*6 + v[2]*6 + v[3];
    assert_eq!(i, 2*5*4*6/2+26);

    let v = t.index_to_position(2*5*4*6-1);
    let i = v[0]*5*4*6 + v[1]*4*6 + v[2]*6 + v[3];
    assert_eq!(i, 2*5*4*6-1);

    let v = t.index_to_position(2*5*4*6);
    assert!(v.is_empty());
}

#[test]
fn tensor_index_test () {
    let xs = (0usize..(3*6*4*2)).map(|i| i as f64).collect::<Vec<f64>>();
    let mut x = Tensor::<f64>::from_vector(vec![3,6,4,2], xs);

    let index = vec![2,3,3,0];
    assert_eq!(x[index.clone()], (index[0]*6*4*2+index[1]*4*2+index[2]*2+index[3]) as f64);

    let index = vec![2,3,2,0];
    x[index.clone()] = 2.0;

    assert_eq!(x[index.clone()], 2.0);
}

#[test]
fn tensor_add_test(){
    let shape:[usize;2] = [3,4];
    let t0 = Tensor::<f32>::zero(&shape);
    let m_init:[f32;12] = [ 11.0,12.0,13.0,14.0,
			    21.0,22.0,23.0,24.0,
			    31.0,32.0,33.0,34.0 ];
    let t1 = Tensor::<f32>::from_array(&[3,4],&m_init);

    let t2 = t1 + t0;
    println!("{}", t2);
}

#[test]
fn subtensor_test() {
    {
	let v = (0..5*6*4).map(|i| i as f64).collect::<Vec<f64>>();
	let t0 = Tensor::<f64>::from_vector(vec![5,6,4], v);
	let st2 = t0.sub_tensor(2);
	println!("{}", t0);
	println!("{}", st2);

	if let Some(u) = t0.position_to_index(&[4,3]){
	    assert_eq!(u,108);
	}
	else {
	    panic!("test failed. {}:{}", file!(), line!());
	}

	assert_eq!(None, t0.position_to_index(&[4,10]));

	if let Some(u) = t0.position_to_index(&[0,0,0]) {
	    assert_eq!(u,0);
	}
	else {
	    panic!("test failed. {}:{}", file!(), line!());
	}

	if let Some(u) = t0.position_to_index(&[1,1,1]) {
	    assert_eq!(u,29);
	}
	else {
	    panic!("test failed. {}:{}", file!(), line!());
	}

	if let Some(u) = t0.position_to_index(&[4]) {
	    assert_eq!(u,96);
	}
	else {
	    panic!("test failed. {}:{}", file!(), line!());
	}
    }

    {
	let v = (0..5*6*4*8).map(|i| i as f64).collect::<Vec<f64>>();
	let t0 = Tensor::<f64>::from_vector(vec![5,6,4,8], v);
	println!("{}", t0);

	if let Some(st) = t0.get_sub_tensor_by_position(&[2]){
	    assert_eq!(st, t0.sub_tensor(2));
	}
	else {
	    panic!("test failed. {}:{}", file!(), line!());
	};

	if None != t0.get_sub_tensor_by_position(&[4,7]) {
	    panic!("test failed. {}:{}", file!(), line!());
	}

	if let Some(st) = t0.get_sub_tensor_by_position(&[2,3]){
	    let st2 = t0.sub_tensor(2).into_tensor();
	    let st23 = st2.sub_tensor(3);
	    assert_eq!(st,st23);
	}

	if let Some(st) = t0.get_sub_tensor_by_position(&[1,3,2]){
	    println!("{}",st);
	    let st1 = t0.sub_tensor(1).into_tensor();
	    let st13 = st1.sub_tensor(3).into_tensor();
	    let st132 = st13.sub_tensor(2);
	    assert_eq!(st,st132);
	}

	if let Some(st) = t0.get_sub_tensor_by_position(&[1,3,2,6]){
	    println!("{}",st);
	    let st1 = t0.sub_tensor(1).into_tensor();
	    let st13 = st1.sub_tensor(3).into_tensor();
	    let st132 = st13.sub_tensor(2).into_tensor();
	    let st1326 = st132.sub_tensor(6);
	    assert_eq!(st,st1326);
	}
    }

}

#[test]
fn add_at_test () {
    {
	let v0 = (0..6*3*7*4).map(|i| i as f64).collect::<Vec<f64>>();
	let t0 = Tensor::<f64>::from_vector(vec![6,3,7,4], v0);
	let t1 = Tensor::<f64>::from_vector(vec![1,4], vec![1.0f64;4]);
	let t2 = t0.add_at(&[5,2,4], &t1);

	let mut t3 = Tensor::<f64>::from_vector(vec![6,3,7,4], vec![0.0f64;6*3*7*4]);
	t3[vec![5,2,4,0]] = 1.0;
	t3[vec![5,2,4,1]] = 1.0;
	t3[vec![5,2,4,2]] = 1.0;
	t3[vec![5,2,4,3]] = 1.0;
	assert_eq!(t2,t3+t0);
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

    {
		let m_init:[f32;3] = [1.0,2.0,3.0];
		let t0 = Tensor::<f32>::from_array(&[1,3], &m_init);
		let t1 = t0.broadcast(&[2,3]);
		println!("{}", t1);
		assert_eq!(t1, Tensor::<f32>::from_array(&[2,3], &[1.0,2.0,3.0,1.0,2.0,3.0]));
		let t2 = t1.broadcast(&[3,2,3]);
		println!("{}", t2);
		assert_eq!(t2, Tensor::<f32>::from_array(&[3,2,3],
												 &[1.0,2.0,3.0,
												   1.0,2.0,3.0,
												   1.0,2.0,3.0,
												   1.0,2.0,3.0,
												   1.0,2.0,3.0,
												   1.0,2.0,3.0]));
    }
}


#[test]
fn tensor_affine_test() {
    let m_init:[f32;12] = [ 11.0,12.0,13.0,14.0,
			    21.0,22.0,23.0,24.0,
			    31.0,32.0,33.0,34.0 ];
    let t1 = Tensor::<f32>::from_array(&[3,4],&m_init);
    let m_init:[f32;20] = [ 11.0,12.0,13.0,14.0,15.0,
			    21.0,22.0,23.0,24.0,25.0,
			    31.0,32.0,33.0,34.0,35.0,
			    41.0,42.0,43.0,44.0,45.0 ];
    let t2 = Tensor::<f32>::from_array(&[4,5],&m_init);
    let t3 = Tensor::<f32>::matrix_product(&t1, &t2);
    assert_eq!(t3[vec![0,0]], 11.0*11.0+12.0*21.0+13.0*31.0+14.0*41.0);
    assert_eq!(t3[vec![1,0]], 21.0*11.0+22.0*21.0+23.0*31.0+24.0*41.0);
    assert_eq!(t3[vec![0,1]], 11.0*12.0+12.0*22.0+13.0*32.0+14.0*42.0);

    let m_init:[f32;9] = [ 2.0,0.0,1.0,
			   2.0,2.0,1.0,
			   3.0,0.0,1.0 ];
    let a = Tensor::<f32>::from_array(&[3,3],&m_init);
    let m_init:[f32;3] = [ 1.0, 2.0, 3.0 ];
    let v = Tensor::<f32>::from_array(&[3,1],&m_init);
    let m_init:[f32;3] = [ 2.0, 2.0, 2.0 ];
    let b = Tensor::<f32>::from_array(&[3,1],&m_init);

    let av = Tensor::<f32>::matrix_product(&a, &v);

    let av_result = [2.0*1.0+0.0*2.0+1.0*3.0,
		     2.0*1.0+2.0*2.0+1.0*3.0,
		     3.0*1.0+0.0*2.0+1.0*3.0];
    assert_eq!(av,Tensor::<f32>::from_array(&[3,1],&av_result));

    let c = Tensor::<f32>::affine(&a,&v,&b);
    assert_eq!(c,av+b);

    let m_init:[f32;3] = [ 6.0,2.0,1.0 ];
    let a = Tensor::<f32>::from_array(&[1,3],&m_init);
    let m_init:[f32;3] = [ 3.0,4.0,1.0 ];
    let v = Tensor::<f32>::from_array(&[3,1],&m_init);
    let av = Tensor::<f32>::matrix_product(&a, &v);

    assert_eq!(av,Tensor::<f32>::from_array(&[1,1],&[3.0*6.0+2.0*4.0+1.0*1.0]));
}
