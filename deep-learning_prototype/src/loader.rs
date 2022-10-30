use std::mem;
use std::fs::File;
use std::result::Result;
use std::io::{Read,BufReader,BufRead,Seek,SeekFrom};
use std::path::Path;
use std::collections::HashMap;
use hdf5;
use byteorder::{ByteOrder, BigEndian};
use linear_transform::tensor::tensor_base::Tensor;

pub trait Loader<T> {
    fn from_hdf5(file:&str) -> Result<HashMap<String,T>,String>;
    fn from_mnist(labels_file: &str, images_file: &str, flatten: bool, normalize: bool) -> Result<(Vec<u32>,Vec<T>),String>;
}

impl<T> Loader<Tensor<T>> for Tensor<T>
    where T: num::Float+num::FromPrimitive+Clone+hdf5::H5Type
{
    fn from_hdf5(file:&str) -> Result<HashMap<String,Tensor<T>>,String> {
	let datasets_file_path = Path::new(file);
	let mut datasets = HashMap::<String,Tensor<T>>::new();

	if !datasets_file_path.exists() {
            return Err("weight file does not exists".to_string());
	}

	let f = match hdf5::File::open(file) {
	    Ok(f) => f,
	    Err(e) => {
		match e {
		    hdf5::Error::HDF5(es) => {
			if let Ok(expanded_es) = es.expand() {
			    return Err(expanded_es.description().to_string());
			}
			else {
			    return Err("Internal Error".to_string());
			}
		    },
		    hdf5::Error::Internal(es) => {
			return Err(es);
		    }
		}
	    }
	};

	let hdf5_datasets = match f.datasets() {
	    Ok(ds) => ds,
	    Err(e) => {
		match e {
		    hdf5::Error::HDF5(es) => {
			if let Ok(expanded_es) = es.expand() {
			    return Err(expanded_es.description().to_string());
			}
			else {
			    return Err("Internal Error".to_string());
			}
		    },
		    hdf5::Error::Internal(es) => {
			return Err(es);
		    }
		}
	    }
	};

	for dataset in hdf5_datasets {
	    let name = dataset.name().clone();
	    println!("{}", dataset.name());
	    if let Ok(values) = dataset.read_raw::<T>() {
		let shape = dataset.shape();
		let t:Tensor<T> = Tensor::<T>::from_vector(shape.to_vec(), values);
		println!("{:?}", shape);
		datasets.insert(name.trim_start_matches('/').to_string(), t);
	    }
	}

	Ok(datasets)
    }

    fn from_mnist(labels_file: &str, images_file: &str, flatten: bool, normalize: bool) -> Result<(Vec<u32>,Vec<Tensor<T>>),String> {
	let mut labels:Vec<u32> = Vec::new();
	let mut images:Vec<Tensor<T>> = Vec::new();
	let labels_file_path = Path::new(labels_file);
	let images_file_path = Path::new(images_file);

	if !labels_file_path.exists() {
	    return Err("minst labels file does not exists".to_string());
	}

	if !images_file_path.exists() {
	    return Err("minst images file does not exists".to_string());
	}

	let mut labels_f = BufReader::new(
	    File::open(labels_file).expect("Failed to open file")
	);

	let mut images_f = BufReader::new(
	    File::open(images_file).expect("Failed to open file")
	);

	let mut header:[u8;8] = unsafe{ mem::MaybeUninit::uninit().assume_init() };
	let num_of_labels = match labels_f.read(&mut header) {
	    Ok(readsize) => {
		if readsize == 8 {
		    let magic_bytes = &header[0..4];
		    let num_bytes = &header[4..8];
		    let magic = BigEndian::read_u32(magic_bytes);
		    let num_of_images = BigEndian::read_u32(num_bytes);
		    if magic != 0x00000801 {
			return Err("labels file magic number is invalid".to_string());
		}
		    num_of_images
		}
		else {
		    return Err("labels file is too short length".to_string());
		}
	    },
	    Err(err) => {
		return Err(err.to_string());
	    }
	};

	for i in 0..num_of_labels {
	    let mut label = [0u8;1];
	    match labels_f.read(&mut label) {
		Ok(readsize) => {
		if readsize == 1 {
		    labels.push(label[0].into());
		}
		else {
		    return Err("labels file is too short length".to_string());
		}
	    },
	    Err(e) => {
		return Err(e.to_string());
	    }
	    }
	}

	let mut header:[u8;16] = unsafe{ mem::MaybeUninit::uninit().assume_init() };
	let (num_of_images,(row,col)) = match images_f.read(&mut header) {
	    Ok(readsize) => {
		if readsize == 16 {
		    let magic = BigEndian::read_u32(&header[0..4]);
		    let num_of_images = BigEndian::read_u32(&header[4..8]);
		    let row = BigEndian::read_u32(&header[8..12]);
		    let col = BigEndian::read_u32(&header[12..16]);
		    if magic != 0x00000803 {
			return Err("images file magic number is invalid".to_string());
		    }
		    (num_of_images, (row, col))
		}
		else {
		    return Err("images file is too short length".to_string());
		}
	    },
	    Err(e) => { return Err(e.to_string()); }
	};

	if num_of_images != num_of_labels {
	    return Err("number of images is different from number of labels".to_string());
	}

	for n in 0..num_of_images {
	    let image_size = (row*col) as u64;
	    let mut image = images_f.by_ref().take(image_size);
	    let bytes = match image.fill_buf() {
		Ok(b) => {
		    if b.len() != image_size as usize {
			return Err("images file is too short length".to_string());
		    }
		    b
		},
		Err(e) => {
		    return Err(e.to_string());
		}
	    };

	    let floats = bytes.iter().map(|x| num::FromPrimitive::from_u8(*x).unwrap()).collect::<Vec<T>>();
	    let shape:[usize;2] = if flatten {
		[1, image_size as usize]
	    }
	    else {
		[col as usize, row as usize]
	    };

	    let m = Tensor::<T>::from_array(&shape, &floats.into_boxed_slice());
	    if normalize {
		let scale:T = num::FromPrimitive::from_f64(1.0/255.0).unwrap();
		images.push(m.scale(scale));
	    }
	    else {
		images.push(m);
	    }

	    if let Err(e) = images_f.seek(SeekFrom::Current(image_size as i64)) {
		return Err(e.to_string());
	    };
	}

	Ok((labels,images))
    }
}
