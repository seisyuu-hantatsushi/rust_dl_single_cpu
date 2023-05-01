use crate::*;
use std::rc::Rc;
use std::ptr;

pub struct Context {
    pub handle: cudnnHandle_t
}

pub type SharableContext = Rc<Context>;

impl Context {
    pub fn new() -> Result<Context, Box<dyn std::error::Error>> {
	let mut handle = ptr::null_mut();
	let result = unsafe { cudnnCreate(&mut handle) };
	if result == cudnnStatus_t_CUDNN_STATUS_SUCCESS {
	    Ok(Context {
		handle
	    })
	}
	else {
	    Err(Box::<CUDNNStatus>::new(result.into()))
	}
    }
}

impl Drop for Context {
    fn drop(&mut self) {
	unsafe { cudnnDestroy(self.handle) };
    }
}
