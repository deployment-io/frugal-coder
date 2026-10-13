use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
pub struct ModelDir(pub PathBuf);
impl ModelDir {
    pub fn new() -> Self {
        let mut attempt = 0;
        let path = loop {
            let path = std::env::temp_dir().join(format!(
                "frugal-m0-test-{}-{}-{attempt}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => break path,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    attempt += 1;
                }
                Err(e) => panic!("create temporary test directory: {e}"),
            }
        };
        let dir = Self(path);
        dir.write(
            "config.json",
            br#"{"model_type":"unknown_synthetic","architectures":["Synthetic"]}"#,
        );
        dir.weights(
            br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &[0; 4],
        );
        dir
    }
    pub fn path(&self) -> &Path {
        &self.0
    }
    pub fn write(&self, name: &str, data: &[u8]) {
        fs::write(self.0.join(name), data).unwrap();
    }
    pub fn weights(&self, header: &[u8], payload: &[u8]) {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(payload);
        self.write("model.safetensors", &bytes);
    }
}
impl Drop for ModelDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
