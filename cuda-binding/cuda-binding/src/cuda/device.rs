use crate::*;

pub struct Device {
    device: CUdevice
}

impl Device {

    pub fn getCount() -> Result<i32, CUDAError> {
	let mut count:i32 = 0;
	let result:CUDAError = unsafe { cuDeviceGetCount(&mut count) }.into();
	if result != CUDAError::Success {
	    return Err(result);
	}
	Ok(count)
    }

    pub fn new(deviceNo:i32) -> Result<Device, CUDAError> {
	let mut device:CUdevice = 0;

	let result:CUDAError = unsafe { cuDeviceGet(&mut device, deviceNo) }.into();
	if result != CUDAError::Success {
	    return Err(result);
	}
	Ok(Device {
	    device
	})
    }

    pub fn as_raw(&self) -> i32 {
	self.device
    }

    pub fn getComputeCapability(&self) -> Result<(u32,u32),CUDAError> {
	let (mut major, mut minor):(i32,i32) = (0,0);
	let result:CUDAError = unsafe {
	    cuDeviceGetAttribute(&mut major,
				 CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
				 self.device) }.into();
	if result != CUDAError::Success {
	    return Err(result);
	}
	let result:CUDAError = unsafe {
	    cuDeviceGetAttribute(&mut minor,
				 CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
				 self.device) }.into();
	if result != CUDAError::Success {
	    return Err(result);
	}
	Ok((major as u32, minor as u32))
    }

    pub fn get_name(&self) -> Result<String, CUDAError> {
	let mut name:[i8;100] = [0;100];
	let result:CUDAError = unsafe {
	    cuDeviceGetName(name.as_mut_ptr(), 100, self.device)
	}.into();
	if result != CUDAError::Success {
	    return Err(result);
	}
	Ok(String::from_utf8(name.into_iter().map(|c| c as u8).collect()).unwrap())
    }
}
