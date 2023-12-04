use core::time::Duration;
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use rand::Rng;
use std::thread::sleep;

lazy_static::lazy_static! {
    pub static ref ALL_GPUS: Vec<AtomicUsize> = init_all_gpus((get_gpu_list()).len(), get_gpu_memory());
}

pub fn get_env_switch(name: String, def: bool) -> bool {
    let res = if let Ok(switch) = env::var(name.clone()) {
        if switch == "true" {
            true
        } else if switch == "false" {
            false
        }
        else {
            def
        }
    } else {
        def
    };
    res
}

fn get_list_from_string(value: String) -> Result<Vec<usize>, serde_json::Error> {
    let res: Vec<usize> = serde_json::from_str(&value)?;
    Ok(res)
}

pub fn get_gpu_list() -> Vec<usize>
{
    let env_key = "OPTION6";
    let res = if let Ok(option_value) = env::var(env_key) {
        if let Ok(gpu_list) = get_list_from_string(option_value) {
            gpu_list
        } else {
            vec![0, 1, 2, 3]
        }
    } else {
        vec![0, 1, 2, 3]
    };
    res
}

fn get_numeric_env(env_key: String, def: usize) -> usize {
    let res = if let Ok(num) = env::var(env_key) {
        if let Ok(num) = num.parse() {
            num
        } else {
            def
        }
    } else {
        def
    };
    res
}

pub fn get_gpu_memory() -> usize
{
    get_numeric_env("OPTION7".to_string(), 10500)
}

pub fn init_all_gpus(gpu_num: usize, _gpu_mem: usize) -> Vec<AtomicUsize> {
    let mut all_gpus= Vec::new();
    for _i in 0..gpu_num {
        all_gpus.push(AtomicUsize::new(181));
    }
    all_gpus
}

pub fn get_all_gpus_consumed_mem() -> Vec<usize> {
    let mut all_gpus_consumed = Vec::new();
    for i in 0..(get_gpu_list()).len() {
        all_gpus_consumed.push((*ALL_GPUS)[i].load(Ordering::SeqCst));
    }
    all_gpus_consumed
}

pub fn min_index(array: &[usize]) -> (usize, usize) {
    let mut min_gpu_idx = 0;
    let mut min_gpu_id = (get_gpu_list())[0];
    let mut min_gpu_mem = array[0];

    for i in 1..(get_gpu_list()).len() {
        if min_gpu_mem>array[i] {
            min_gpu_idx = i;
            min_gpu_id = (get_gpu_list())[i];
            min_gpu_mem = array[i];
        }
    }

    (min_gpu_idx, min_gpu_id)
}

pub fn get_gpu(consume_mem: usize) -> (usize, usize)
{
    let mut rng = rand::thread_rng();
    loop {
        let rand_sleep = rng.gen_range(0..10);
        sleep(Duration::from_millis(rand_sleep));
        let (gpu_ok, min_idx, gpu_id) = free_gpu_ok(consume_mem);
        if gpu_ok {
            return (min_idx, gpu_id);
        }else {
            let sleep_time = Duration::from_millis(1000);
            sleep(sleep_time);
        }
    }
}
pub fn free_gpu_ok(consume_mem: usize) -> (bool, usize, usize) {
    let all_gpus_consumed = get_all_gpus_consumed_mem();
    let (min_idx, gpu_id) = min_index(&all_gpus_consumed);
    // println!("gpu memory: {}, consume_mem: {}", get_gpu_memory(), consume_mem);
    if (*ALL_GPUS)[min_idx].load(Ordering::SeqCst) < (get_gpu_memory()) - consume_mem {
        let _ = (*ALL_GPUS)[min_idx].fetch_add(consume_mem, Ordering::SeqCst);
        // println!("device idx {:}, gpu_id: {} used gpu memory: {:?}", min_idx, gpu_id, new_used_mem+consume_mem);
        (true, min_idx, gpu_id)
    } else {
        (false, min_idx, gpu_id)
    }
}

pub fn finish_use_gpu(min_idx: usize, consume_mem: usize) {
    (*ALL_GPUS)[min_idx].fetch_sub(consume_mem, Ordering::SeqCst);
}