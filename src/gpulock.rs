use std::sync::atomic::{AtomicU64, AtomicU32, AtomicU8, AtomicBool, Ordering};
use crossbeam_channel::{unbounded, Sender, Receiver};
use crossbeam_queue::SegQueue;
use std::fmt::Debug;
use rand::thread_rng;
use rand::prelude::SliceRandom;
use std::{thread, time::Duration};

lazy_static::lazy_static! {
    pub static ref LOCKER_1GPU :GpuLock = GpuLock::new(4, 11000, 800); // Test Purpose
    pub static ref LOCKER_4GPU :GpuLock = GpuLock::new(4, 11000, 0); // Test Purpose
    pub static ref FFT_MEM :FftInstance = FftInstance::default();
    pub static ref MSM_MEM :MsmInstance = MsmInstance::default();

}

pub static CPU_IDX: usize = 999;

pub trait GpuInstance: Copy + Debug + Send + Sync {
    fn get_mem(&self, k: u32) -> u32;
}

#[derive(Clone, Debug, Copy)]
pub struct FftInstance {
    baseline: u32,
}

impl FftInstance {
    pub fn default() -> Self {
        FftInstance {
            baseline: 1200,
        }
    }
}

impl GpuInstance for FftInstance {
    fn get_mem(&self, k: u32) -> u32 {
        match k {
            24 => self.baseline,
            25..=26 => self.baseline<<(k-24),
            20..=23  => (self.baseline>>(24-k)),
            _ => 10
        } 
    }
}


#[derive(Clone, Debug, Copy)]
pub struct MsmInstance {
    baseline: u32,
}

impl MsmInstance {
    pub fn default() -> Self {
        MsmInstance {
            baseline: 3000,
        }
    }
}

impl GpuInstance for MsmInstance {
    fn get_mem(&self, k: u32) -> u32 {
        match k {
            24 => self.baseline,
            25..=26 => self.baseline<<(k-24),
            20..=23  => (self.baseline>>(24-k)),
            _ => 10
        } 
    }
}

pub struct GpuLock { 
    gpu_num: u8,
    mem_total: u32,
    mem_used: Vec<AtomicU32>,
    sender: Sender<u64>,
    receiver: Receiver<u64>,
    current_tasks: AtomicU64,
    pending_tasks: AtomicU64,
    running_tasks: Vec<AtomicU8>,
    queued_tasks_num: AtomicU64,
    queued_tasks: SegQueue<u64>,
    cpu_lock: AtomicBool,
}



impl GpuLock {
    pub fn new(gpu_num: u8, mem_total: u32, mem_used: u32) -> Self {
        let (sender, receiver) = unbounded();
        GpuLock {
            gpu_num,
            mem_total, 
            mem_used: (0..gpu_num).map(|_| AtomicU32::new(mem_used)).collect(),
            sender,
            receiver,
            current_tasks: AtomicU64::new(0),
            pending_tasks: AtomicU64::new(0),
            running_tasks: (0..gpu_num).map(|_| AtomicU8::new(0)).collect(),
            queued_tasks_num: AtomicU64::new(0),
            queued_tasks: SegQueue::new(),
            cpu_lock: AtomicBool::new(false),
        }
    }

    pub fn default() -> Self {
        let (sender, receiver) = unbounded();
        GpuLock {
            gpu_num: 1,
            mem_total: 11000, 
            mem_used: vec![AtomicU32::new(0)],
            sender,
            receiver,
            current_tasks: AtomicU64::new(0),
            pending_tasks: AtomicU64::new(0),
            running_tasks: vec![AtomicU8::new(0)],
            queued_tasks_num: AtomicU64::new(0),
            queued_tasks: SegQueue::new(),
            cpu_lock: AtomicBool::new(false),

        }
    }

    pub fn is_full(&self, required_mem :u32) -> bool {
        for mem_used in self.mem_used.iter() {
            if self.mem_total > mem_used.load(Ordering::SeqCst)+required_mem {
                return false;
            }
        }
        return true;
    }

    pub fn acquire_gpu(&self, required_mem: u32) -> usize {
        let task_id = self.queued_tasks_num.fetch_add(1, Ordering::SeqCst);
        let mut rng: rand::rngs::ThreadRng = rand::thread_rng();
        if !self.queued_tasks.is_empty() {
            let mut rng: rand::rngs::ThreadRng = rand::thread_rng();
            thread::sleep(Duration::from_millis(50));
        }
        let mut passed = false;

        // while self.pending_tasks.load(Ordering::SeqCst) > u64::from(self.gpu_num * 8) {
        //     self.receiver.recv().unwrap();
        // }
        //log::info!("acquire_gpu, task id is {:?}, pending_tasks is {:?}", task_id, self.pending_tasks.load(Ordering::SeqCst));
        //log::info!("GPU usage:{:?}, running tasks {:?}", self.mem_used, self.running_tasks);
        while true {
            if !self.cpu_lock.load(Ordering::SeqCst) {
                self.cpu_lock.store(true, Ordering::SeqCst);
                if passed {
                    self.pending_tasks.fetch_sub(1, Ordering::SeqCst);
                }
                // if self.current_tasks.load(Ordering::SeqCst) % 100 == 0 {
                //     log::info!("{:?} jobs has been sent to GPU", self.current_tasks.load(Ordering::SeqCst));
                // }
                return CPU_IDX;
            }

            let mut gpu_indice = (0..usize::from(self.gpu_num)).collect::<Vec<usize>>();
            gpu_indice.shuffle(&mut thread_rng());


            let mut min_task_num = 24;
            let mut min_idx = 0;

            for gpu_idx in gpu_indice.iter() {
                let current_task_num = self.running_tasks[*gpu_idx].load(Ordering::SeqCst);
                if current_task_num < min_task_num {
                    min_task_num = current_task_num;
                    min_idx = *gpu_idx;
                }

            }
            if self.mem_total > self.mem_used[min_idx].load(Ordering::SeqCst) {
                self.mem_used[min_idx].fetch_add(required_mem, Ordering::SeqCst);
                self.current_tasks.fetch_add(1, Ordering::SeqCst);
                self.running_tasks[min_idx].fetch_add(1, Ordering::SeqCst);
                if passed {
                    self.pending_tasks.fetch_sub(1, Ordering::SeqCst);
                }
                // if self.current_tasks.load(Ordering::SeqCst) % 100 == 0 {
                //     log::info!("{:?} jobs has been sent to GPU", self.current_tasks.load(Ordering::SeqCst));
                // }
                //log::info!("GPU {:?} is in use", min_idx);
                return min_idx;
            }
            
            if !passed {
                self.queued_tasks.push(task_id);

                let mut next_task_id = self.receiver.recv().unwrap();
                while !(((next_task_id > task_id) || next_task_id == 0) && self.pending_tasks.load(Ordering::SeqCst) < 4)  {
                    next_task_id = self.receiver.recv().unwrap();
                }
                passed = true;
                //log::info!("Too add pending task {:?} into pending tasks {:?}", task_id, self.pending_tasks.load(Ordering::SeqCst)); 
                self.pending_tasks.fetch_add(1, Ordering::SeqCst);
            }
            //log::info!("No wait needed, go ahead to loop! for this task {:?}, lopp account is {:?}, wait {:?} MS",task_id, loop_count, 512/loop_count);
            //thread::sleep(Duration::from_millis(512/loop_count));
        }
        return 0;
    }

    pub fn release_gpu(&self, required_mem: u32, gpu_idx: usize) {
        if  gpu_idx == CPU_IDX {
            self.cpu_lock.store(false, Ordering::SeqCst);

        }
        else {        
            self.mem_used[gpu_idx].fetch_sub(required_mem, Ordering::SeqCst);
            self.running_tasks[gpu_idx].fetch_sub(1, Ordering::SeqCst);
        }
        //log::info!("GPU lock released, running jobs are {:?}, memory used are {:?} for GPU_{:?}, subbed memory are {:?} ", self.running_tasks, self.mem_used[gpu_idx].load(Ordering::SeqCst),  gpu_idx,required_mem);
        //log::info!("Channel length is {:?}", self.sender.len());
        if !self.queued_tasks.is_empty() {
            let msg = self.queued_tasks.pop().unwrap();
            self.sender.send(msg).unwrap();
            //log::info!("Finished GPU job trigger task {:?} into loop", msg);

        } 
        else {
            self.sender.send(0).unwrap();
        }
        
    }

    pub fn get_used(&self, gpu_idx: usize) -> u32 {
        self.mem_used[gpu_idx].load(Ordering::SeqCst)
    }

    pub fn get_current_tasks(&self) -> u64 {
        self.current_tasks.load(Ordering::SeqCst)

    }

    pub fn get_running_tasks(&self) -> u8 {
        let mut ret = 0;
        self.running_tasks.iter().for_each(|jobs| ret += jobs.load(Ordering::SeqCst));
        ret
    }

    pub fn get_gpu_num(&self) -> u8 {
        self.gpu_num
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::threadpool::{Worker, Waiter};
    use rand::prelude::SliceRandom;
    use rand::Rng;
    use std::{thread, time::Duration};

    fn simulate_computing() -> usize {
        let fft_mem: u32 = FFT_MEM.get_mem(7);
        let msm_mem = 3300;
        let what_mem = 6700;

        let mem_requirements: Vec<u32> = vec![fft_mem, msm_mem, what_mem];
        
        let mut rng = rand::thread_rng();
        let mem_required = mem_requirements.choose(&mut rand::thread_rng()).unwrap().clone();
        let gpu_idx = LOCKER_4GPU.acquire_gpu(mem_required);
        thread::sleep(Duration::from_millis(rng.gen_range(1000..5000)));
        LOCKER_4GPU.release_gpu(mem_required, gpu_idx);
        gpu_idx
    }

    
    #[test]
    fn test_gpulock() {

        use env_logger::Builder;
        use log::LevelFilter;

        Builder::new().filter(None, LevelFilter::Debug).parse_default_env().init();
        let worker = Worker::new();

        let task_num = 100;
        
        let rets : Vec<Waiter<_>> = (0..task_num).map(|_| {
            worker.compute(move || simulate_computing())
        }).collect();

        let rets: Vec<usize> = rets.iter().enumerate().map(|(i, ret)| {
            if i % 10 == 0 {
                log::info!("Proceeded {:?} gpu threads", i);
            }
            ret.wait()
        }).collect();
        log::info!("Test finished, total tasks are: {:?}, running tasks are: {:?}", LOCKER_4GPU.get_current_tasks(), LOCKER_4GPU.get_running_tasks());
        let mut i = 0;
        while i < LOCKER_4GPU.get_gpu_num() {
            log::info!("Available memory for gpu {:?} is {:?}", i, LOCKER_4GPU.get_used(i as usize));
            assert_eq!(LOCKER_4GPU.get_used(i as usize), 0);
            i += 1;
        }
        assert_eq!(LOCKER_4GPU.get_running_tasks(), 0);

    }

    #[test]
    fn test_fft_mem() {
        assert_eq!(FFT_MEM.get_mem(24), 1200);
        assert_eq!(FFT_MEM.get_mem(23), FFT_MEM.get_mem(24)/2);
        assert_eq!(FFT_MEM.get_mem(25), FFT_MEM.get_mem(24)*2);
    }
}