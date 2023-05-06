use crate::*;
use super::*;

pub fn initailize() -> CUDAError {
    let result = unsafe {
	cuInit(0)
    };
    result.into()
}
