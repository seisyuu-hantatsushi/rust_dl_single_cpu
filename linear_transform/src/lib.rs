pub mod vector;
pub mod matrix;

#[cfg(test)]
mod tests {
    use super::*;
    use vector::Vector2;
    use vector::Vector3;
    use matrix::Matrix4x4;
    use matrix::MatrixMxN;

    #[test]
    fn vector2_add(){
	let a = Vector2(1.0, 2.0);
	let b = Vector2(2.0, 1.0);
	assert_eq!(a+b, Vector2(3.0,3.0));
    }

    #[test]
    fn vector2_sub(){
	let a = Vector2(1.0, 2.0);
	let b = Vector2(2.0, 1.0);
	assert_eq!(a-b, Vector2(-1.0,1.0));
    }

    #[test]
    fn vector2_mul(){
	let a = Vector2(1.0, 2.0);
	let b = Vector2(2.0, 1.0);
	assert_eq!(a*b, b*a);
    }

    #[test]
    fn vector2_length(){
	let id = Vector2(3.0, 4.0);
	assert_eq!(id.square(), 25.0);
	assert_eq!(id.length(), 5.0);
    }

    #[test]
    fn vector3_cross_product(){
	let a = Vector3(1.0, 2.0, 3.0);
	let b = Vector3(2.0, 3.0, 4.0);
	assert_eq!(Vector3::cross_product(a,b), Vector3(-1.0,2.0,-1.0));
    }

    #[test]
    fn matrix4x4_test() {
	let m1 = Matrix4x4{
	    v: [ [1.0, 1.0, 1.0, 1.0],
		  [2.0, 2.0, 2.0, 2.0],
		  [3.0, 3.0, 3.0, 3.0],
		  [4.0, 4.0, 4.0, 4.0],
	    ]
	};
	let m2 = Matrix4x4 {
	    v: [ [4.0, 4.0, 4.0, 4.0],
		  [3.0, 3.0, 3.0, 3.0],
		  [2.0, 2.0, 2.0, 2.0],
		  [1.0, 1.0, 1.0, 1.0],
	    ]
	};
	let m3 = Matrix4x4 {
	    v: [ [5.0, 5.0, 5.0, 5.0],
		  [5.0, 5.0, 5.0, 5.0],
		  [5.0, 5.0, 5.0, 5.0],
		  [5.0, 5.0, 5.0, 5.0] ]
	};

	let m4 = Matrix4x4{
	    v: [  [11.0, 12.0, 13.0, 14.0],
		  [21.0, 22.0, 23.0, 24.0],
		  [31.0, 32.0, 33.0, 34.0],
		  [41.0, 42.0, 43.0, 44.0],
	    ]
	};
	let m5 = Matrix4x4 {
	    v: [  [1.0, 2.0, 3.0,  4.0],
		  [2.0, 4.0, 6.0,  8.0],
		  [3.0, 6.0, 12.0, 16.0],
		  [4.0, 8.0, 15.0, 20.0],
	    ]
	};
	let m6 = Matrix4x4 {
	    v: [ [130.0, 260.0, 471.0,  628.0],
		 [230.0, 460.0, 831.0,  1108.0],
		 [330.0, 660.0, 1191.0, 1588.0],
		 [430.0, 860.0, 1551.0, 2068.0] ]
	};

	assert_eq!(m1+m2, m3);
	assert_eq!(m3-m1, m2);
	assert_eq!(m1*Matrix4x4::identity(), Matrix4x4::identity()*m1);
	assert_eq!(m4*m5, m6);
    }

    #[test]
    fn matrix_nxm_test(){
	{
	    let mut m1 = MatrixMxN::zero(28,28);
	    assert_eq!(m1.shape(),(28,28));
	    for i in 0..28 {
		for j in 0..28 {
		    m1[i][j] = (i as f64)*100.0+(j as f64);
		}
	    }
	    assert_eq!(m1[0][0],0.0);
	    assert_eq!(m1[1][0],100.0);
	    assert_eq!(m1[0][1],1.0);
	    assert_eq!(m1[2][2],202.0);
	    assert_eq!(m1[26][27],2627.0);
	    assert_eq!(m1[27][26],2726.0);
	    assert_eq!(m1[27][27],2727.0);


	    let mut m2 = MatrixMxN::zero(6,5);
	    assert_eq!(m2.shape(),(6,5));
	    for i in 0..6 {
		for j in 0..5 {
		    m2[i][j] = (i as f64)*100.0+(j as f64);
		}
	    }

	    assert_eq!(m2[1][2],102.0);

	    let m3 = m2.transpose();
	    assert_eq!(m3.shape(),(5,6));
	    assert_eq!(m3[1][2],201.0);
	    assert_eq!(m3[2][2],202.0);

	    let m4 = m2.clone() * m3.clone();
	    {
		let mut v:f64 = 0.0;

		for k in 0..5 {
		    v += m2[1][k]*m3[k][2];
		}

		assert_eq!(m4[1][2],v);
	    }

	    {
		let mut v:f64 = 0.0;

		for k in 0..5 {
		    v += m2[3][k]*m3[k][2];
		}

		assert_eq!(m4[3][2],v);
	    }
	}
	{
	    let mut m1 = MatrixMxN::zero(1,28);
	    let mut m2 = MatrixMxN::zero(28,1);
	    for k in 0..28 {
		m1[0][k] = k as f64;
		m2[k][0] = k as f64;
	    }
	    let m3 =  MatrixMxN::mul(&m1, &m2);

	    {
		let mut v:f64 = 0.0;

		for k in 0..28 {
		    v += m1[0][k]*m2[k][0];
		}

		assert_eq!(m4[0][0],v);
	    }

	    {
		let mut v:f64 = 0.0;

		for k in 0..28 {
		    v += m1[0][k]*m2[k][0];
		}

		assert_eq!(m4[0][0],v);
	    }
	}
    }

}
