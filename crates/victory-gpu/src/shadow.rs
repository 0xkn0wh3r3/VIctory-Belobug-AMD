//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// The copy's bytes before the submission ran, from an offset on,
    /// covering the ranges its shaders write: what the GPU wrote differs.
    before: Mutex<Before>,
    /// The buffer is the title's memory itself ([`direct`]): nothing is
    /// copied either way.
    direct: Option<Direct>,
    /// Ranges that cannot be watched, with the frame they were copied in.
    unwatched: Mutex<std::collections::HashMap<(usize, usize), u64>>,
    /// The buffer has memory only where it was brought up to date.
    sparse: Option<Sparse>,
}

struct Direct {
    heap: windows::Win32::Graphics::Direct3D12::ID3D12Heap,
    /// The same memory from other 64 KiB starts, for views whose elements
    /// do not fall on multiples of their size from this one's.
    aliases: Mutex<Vec<(u64, Arc<Shadow>)>>,
    /// The title mapped other memory there since.
    dead: std::sync::atomic::AtomicBool,
}

#[derive(Default)]
struct Before {
    snapshot: Option<(usize, Vec<u8>)>,
    unchanged: UnchangedPages,
}

/// Pages already compared in full against this submission's snapshot.
#[derive(Default)]
struct UnchangedPages {
    start: usize,
    end: usize,
    pages: Vec<bool>,
}

impl UnchangedPages {
    fn begin(&mut self, start: usize, end: usize) {
        if self.pages.is_empty() {
            self.start = start;
            self.end = end;
            self.pages.resize(end.div_ceil(PAGE) - start / PAGE, false);
        }
    }

    fn differs(&mut self, address: usize, current: &[u8], before: &[u8]) -> bool {
        let page = address / PAGE;
        let unchanged = &mut self.pages[page - self.start / PAGE];
        if *unchanged {
            return false;
        }
        if current != before {
            return true;
        }
        // A later range can touch bytes a partial comparison missed. At
        // the edges, only bytes covered by the snapshot can be compared.
        *unchanged = address == (page * PAGE).max(self.start)
            && address + current.len() == ((page + 1) * PAGE).min(self.end);
        false
    }

    fn clear(&mut self) {
        self.pages.clear();
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Bytes of the snapshot memory kept for reuse.
pub(crate) fn pool_bytes() -> u64 {
    let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.iter().map(|v| v.capacity() as u64).sum()
}

/// Bytes the pool keeps at most: snapshots of the world's 174 MB heap
/// would otherwise keep 8 of that size (1.4 GB of host memory) for good.
const POOL_BYTES: usize = 512 << 20;

/// Returns a snapshot buffer to the pool (at most 8 kept, the largest,
/// and no more than [`POOL_BYTES`] in all).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    pool.truncate(8);
    let mut total = 0;
    pool.retain(|v| {
        total += v.capacity();
        total <= POOL_BYTES
    });
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Drop for Shadow {
    fn drop(&mut self) {
        if self.direct.is_some() || self.sparse.is_some() {
            return;
        }
        // SAFETY: a live resource.
        let bytes = unsafe { self.buffer.GetDesc() }.Width;
        crate::memstat::sub(&crate::memstat::SHADOWS, bytes);
        crate::memstat::sub(&crate::memstat::SHADOW_COUNT, 1);
    }
}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// The copy's bytes at `[address, address + len)` (inside the buffer).
    #[cfg(test)]
    pub(crate) fn host_bytes(&self, address: u64, len: usize) -> Vec<u8> {
        // SAFETY: inside the mapping; no submission runs while tests read.
        unsafe { std::slice::from_raw_parts(self.data.add(self.offset(address) as usize), len) }.to_vec()
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        if self.direct.is_some() {

            return;
        }
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        // SAFETY: a live resource.
        let bytes = unsafe { buffer.GetDesc() }.Width;
        crate::memstat::add(&crate::memstat::SHADOWS, bytes);
        crate::memstat::add(&crate::memstat::SHADOW_COUNT, 1);
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
            direct: None,
            unwatched: Mutex::default(),
            sparse: None,
        }
    }

    /// Whether the buffer is the title's memory itself.
    pub(crate) fn is_direct(&self) -> bool {
        self.direct.is_some()
    }

    pub(crate) fn is_sparse(&self) -> bool {
        self.sparse.is_some()
    }

    /// The same for copies that share their memory.
    pub(crate) fn root_key(&self) -> usize {
        self.sparse_root()
            .map_or(std::ptr::from_ref(self) as usize, |(r, _)| std::ptr::from_ref(r) as usize)
    }

    /// Whether the title mapped other memory there since.
    pub(crate) fn is_dead(&self) -> bool {
        self.direct
            .as_ref()
            .is_some_and(|d| d.dead.load(Ordering::Acquire))
    }

    /// This memory from a start `address` is a multiple of `element` bytes
    /// after.
    fn on_element(self: &Arc<Self>, address: u64, element: u64) -> Option<Arc<Shadow>> {
        if element == 0 || (address - self.start) % element == 0 {
            return Some(self.clone());
        }
        let direct = self.direct.as_ref()?;
        let (page, first) = (PAGE as u64, self.start);
        let mut base = first + page;
        while base <= address && base - first <= 4096 * page {
            if (address - base) % element == 0 {
                let mut aliases = direct.aliases.lock().unwrap_or_else(|e| e.into_inner());
                if let Some((_, s)) = aliases.iter().find(|(b, _)| *b == base) {
                    return Some(s.clone());
                }
                let made = Arc::new(placed(&direct.heap, base - first, base, first + self.len - base)?);
                aliases.push((base, made.clone()));
                return Some(made);
            }
            base += page;
        }
        None
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        if self.direct.is_some() {
            return;
        }
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, grown to cover every range the
    /// submission declares (the world declares overlapping views of its
    /// 174 MB heap dozens of times), in reused memory (fresh snapshots cost
    /// ~2.3 million page faults per world frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        if self.direct.is_some() {
            return;
        }
        if let Some((root, s)) = self.sparse_root() {
            return root.sparse_writing(s, address, len);
        }
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address + len));
        let (from, to) = self.inside(address, len);
        // SAFETY: inside the mapping.
        let host =
            |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let before = &mut before.snapshot;
        let (start, end) = match &*before {
            Some((at, bytes)) => ((*at).min(from), (at + bytes.len()).max(to)),
            None => (from, to),
        };
        if let Some((at, bytes)) = &*before
            && (*at, at + bytes.len()) == (start, end)
        {
            return;
        }
        // Parts newly covered from the copy, the rest as it was taken.
        let mut grown = pooled(end - start);
        match before.take() {
            Some((at, bytes)) => {
                grown.extend_from_slice(host(start, at));
                grown.extend_from_slice(&bytes);
                grown.extend_from_slice(host(at + bytes.len(), end));
                give_back(bytes);
            }
            None => grown.extend_from_slice(host(start, end)),
        }
        *before = Some((start, grown));
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        if self.direct.is_some() {
            return 0;
        }
        if let Some((root, s)) = self.sparse_root() {
            return root.sparse_back(s, address, len);
        }
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let Before {
            snapshot,
            unchanged,
        } = &mut *before;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if let Some((taken, before)) = &*snapshot
            && *taken <= from
            && to <= taken + before.len()
            && from < to
        {
            unchanged.begin(start + taken, start + taken + before.len());
            let before = &before[from - taken..to - taken];
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            // An unchanged page is never modified by copy-back, so later
            // overlapping ranges can skip it for the rest of the submission.
            let mut title = None;
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PAGE - (start + at) % PAGE).min(to - at);
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                if unchanged.differs(start + at, host, old)
                    && let title = &mut title.get_or_insert_with(|| vec![0u8; PAGE])[..n]
                    && crate::host::read_into((start + at) as *const u8, title)
                {
                    // Title memory with the GPU's bytes put in; the copy
                    // takes the title's bytes in turn.
                    let mut changed = false;
                    for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
                        if *h != *o && *t != *h {
                            *t = *h;
                            changed = true;
                        }
                        *h = *t;
                    }
                    if changed {
                        crate::watch::writing(start + at, n);
                        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                        written += n;
                    }
                    synced.push((start + at) / PAGE);
                }
                at += n;
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if first < last
                && !synced.is_empty()
                && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
                && let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true)
            {
                let base = start / PAGE;
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                    copied[page - base] = now;
                }
            }
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&r| r != (address, address + len));
        if pending.is_empty()
            && let Some((_, bytes)) = snapshot.take()
        {
            unchanged.clear();
            give_back(bytes);
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        self.update_lazily(address, len, false)
    }

    /// [`Shadow::update`]; `lazy`: the tiles of a sparse copy that did not
    /// change the last times are looked at less often.
    pub(crate) fn update_lazily(&self, address: usize, len: usize, lazy: bool) -> usize {
        if self.direct.is_some() {
            return 0;
        }
        if let Some(root) = self.sparse.as_ref().and_then(|s| s.of.as_ref()) {
            return root.update_lazily(address, len, lazy);
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a, lazy);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to, lazy);
            }
            return bytes;
        }
        self.update_free(from, to, lazy)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize, lazy: bool) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        // shortcut: memory that cannot be watched (unmapped, mostly) is copied
        // once a frame.
        let frame = crate::queue::presented() + 1;
        {
            let mut seen = self.unwatched.lock().unwrap_or_else(|e| e.into_inner());
            if !cfg!(test) && seen.insert((from, to), frame) == Some(frame) {
                return 0;
            }
        }
        if let Some(s) = self.sparse.as_ref().filter(|_| !cfg!(test)) {
            return self.copy_tiles(s, from, to, frame, lazy);
        }
        self.copy(from, to);
        to - from
    }

    /// Copies title memory `[from, to)` into the tiles of a sparse copy;
    /// `lazy`: not those whose turn it is not.
    ///
    /// shortcut: a tile that did not change when looked at twice is next
    /// looked at after 2, 4 and at most 16 frames, unless something new reads
    /// it: the title fills memory, then makes views of it.
    fn copy_tiles(&self, s: &Sparse, from: usize, to: usize, frame: u64, lazy: bool) -> usize {
        thread_local! {
            static READ: std::cell::RefCell<Vec<u8>> = std::cell::RefCell::new(vec![0; PAGE]);
        }
        let mut bytes = 0;
        self.each_tile(s, from, to, |at, data, n| {
            if n < PAGE {
                crate::host::copy_mapped(data, at as *const u8, n, false);
                bytes += n;
                return;
            }
            let tile = ((at - self.start as usize) / PAGE) as u32;
            let seen = {
                let looks = s.looks.lock().unwrap_or_else(|e| e.into_inner());
                looks.get(&tile).copied()
            };
            if lazy && seen.is_some_and(|(next, _)| frame < next) {
                return;
            }
            bytes += n;
            // Only written where it changed: the copy is in memory the CPU
            // reads quickly.
            let same = READ.with_borrow_mut(|read| {
                if !crate::host::read_into(at as *const u8, &mut read[..n]) {
                    crate::host::copy_mapped(data, at as *const u8, n, false);
                    return false;
                }
                // SAFETY: the tile's mapping; only submissions write it.
                let host = unsafe { std::slice::from_raw_parts_mut(data, n) };
                let same = *host == read[..n];
                if !same {
                    host.copy_from_slice(&read[..n]);
                }
                same
            });
            let still = match seen {
                Some((_, still)) if same => (still + 1).min(5),
                _ => 0,
            };
            // Not all in the same frame.
            let wait = match still {
                0 | 1 => 1,
                k => (1 << (k - 1)) - u64::from(tile) % 2,
            };
            s.looks
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(tile, (frame + wait, still));
        });
        bytes
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if let Some(s) = &self.sparse {
            self.each_tile(s, from, to, |at, data, n| {
                crate::host::copy_mapped(data, at as *const u8, n, false);
            });
            return to - from;
        }
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        to - from
    }
}

/// Windows of title memory the host GPU reads and writes in place, by start:
/// each a run of committed pages of one allocation.
static DIRECT: Mutex<BTreeMap<u64, Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn windows_of_memory() -> std::sync::MutexGuard<'static, BTreeMap<u64, Arc<Shadow>>> {
    DIRECT.lock().unwrap_or_else(|e| e.into_inner())
}

/// Buffers of windows gone since (by interface address): views of them are
/// made again.
static GONE: Mutex<std::collections::BTreeSet<usize>> = Mutex::new(std::collections::BTreeSet::new());

/// Whether a host buffer is one of a window gone since.
pub(crate) fn gone(buffer: &ID3D12Resource) -> bool {
    use windows::core::Interface;
    let gone = GONE.lock().unwrap_or_else(|e| e.into_inner());
    !gone.is_empty() && gone.contains(&(buffer.as_raw() as usize))
}

/// A buffer over part of a heap.
fn placed(
    heap: &windows::Win32::Graphics::Direct3D12::ID3D12Heap,
    offset: u64,
    start: u64,
    len: u64,
) -> Option<Shadow> {
    use windows::Win32::Graphics::Direct3D12::*;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: len,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: a live heap that holds the range.
    let made = unsafe {
        host()?.device.CreatePlacedResource(
            heap,
            offset,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        )
    };
    if let Err(error) = made {
        tracing::warn!(%error, start = format_args!("{start:#x}"), len, "no buffer over title memory");
        return None;
    }
    let buffer = buffer?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x} in place"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    Some(Shadow {
        buffer,
        data: start as *mut u8,
        gpu,
        start,
        len,
        copied: Mutex::default(),
        watched: AtomicU8::new(NOT_WATCHED),
        pending: Mutex::default(),
        before: Mutex::default(),
        direct: Some(Direct {
            heap: heap.clone(),
            aliases: Mutex::default(),
            dead: std::sync::atomic::AtomicBool::new(false),
        }),
        unwatched: Mutex::default(),
        sparse: None,
    })
}

/// How far a window reaches at most (host heaps end below 4 GiB).
const WINDOW_REACH: u64 = 1 << 30;

/// The run of committed pages of one allocation that holds `[from, to)`, in
/// whole 64 KiB pages.
fn run_of(from: u64, to: u64) -> Option<(u64, u64)> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_IMAGE, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE, VirtualQuery,
    };
    let query = |at: u64| {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: a query of our own address space.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        (n != 0).then_some(info)
    };
    let usable = |i: &MEMORY_BASIC_INFORMATION| {
        i.State == MEM_COMMIT
            && i.Type != MEM_IMAGE
            && matches!(i.Protect, PAGE_READWRITE | PAGE_READONLY)
    };
    let first = query(from).filter(usable)?;
    let allocation = first.AllocationBase as u64;
    let low = (from & !(WINDOW_REACH - 1)).max(allocation);
    let high = to.checked_next_multiple_of(WINDOW_REACH)?;
    // A query gives the region from the address up, not down: from the
    // bottom on.
    let (mut start, mut end) = (low, low);
    while end < high
        && let Some(i) = query(end)
        && i.AllocationBase as u64 == allocation
    {
        let next = i.BaseAddress as u64 + i.RegionSize as u64;
        if !usable(&i) {
            if end > from {
                break;
            }
            start = next;
        }
        end = next;
    }
    let page = PAGE as u64;
    let (start, end) = (start.max(low).next_multiple_of(page), end.min(high) & !(page - 1));
    // Not short views: the title maps and unmaps those (streaming), which
    // the host refuses while the GPU uses them.
    if first.Type == windows_sys::Win32::System::Memory::MEM_MAPPED && end - start < 16 << 20 {
        return None;
    }
    (start <= from && end >= to).then_some((start, end))
}

/// Whether `[address, end)` lies in a window made already.
pub(crate) fn in_place(address: u64, end: u64) -> bool {
    windows_of_memory()
        .range(..=address)
        .rev()
        .take(8)
        .any(|(_, s)| s.start + s.len >= end)
}

/// The window of title memory holding `[address, end)`, made on first use.
/// `None` where the host GPU cannot use the memory in place: parts of it
/// not committed, or of several allocations (views of physical pages).
fn direct(address: u64, end: u64) -> Option<Arc<Shadow>> {
    use windows::Win32::Graphics::Direct3D12::{ID3D12Device13, ID3D12Heap};
    use windows::core::Interface;
    static DEVICE: std::sync::OnceLock<Option<ID3D12Device13>> = std::sync::OnceLock::new();
    if cfg!(test) {
        return None;
    }
    let device = DEVICE.get_or_init(|| {
        // Diagnostics: copies, as without this.
        let on = std::env::var_os("VICTORY_COPY_MEMORY").is_none();
        let device: Option<ID3D12Device13> = if on {
            host().and_then(|h| h.device.cast().ok())
        } else {
            None
        };
        tracing::info!(on = device.is_some(), "title memory used in place by the host GPU");
        device
    });
    let device = device.as_ref()?;
    let mut map = windows_of_memory();
    if let Some(s) = map
        .range(..=address)
        .rev()
        .take(8)
        .map(|(_, s)| s)
        .find(|s| s.start + s.len >= end)
    {
        return Some(s.clone());
    }
    // Asked over and over for the same memory: not looked at each time.
    static REFUSED: Mutex<BTreeMap<u64, (u64, u64)>> = Mutex::new(BTreeMap::new());
    let frame = crate::queue::presented();
    let mut refused = REFUSED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&(e, f)) = refused.get(&address)
        && e == end
        && f + 30 > frame
    {
        return None;
    }
    let page = PAGE as u64;
    let window = run_of(address & !(page - 1), end.next_multiple_of(page)).and_then(|(start, stop)| {
        // Read-only guards would refuse the GPU's writes.
        crate::watch::writing(start as usize, (stop - start) as usize);
        // SAFETY: committed memory; the heap keeps its pages.
        let heap: Result<ID3D12Heap, _> = unsafe {
            device.OpenExistingHeapFromAddress1(start as *const _, (stop - start) as usize)
        };
        match heap {
            Ok(heap) => placed(&heap, 0, start, stop - start),
            Err(error) => {
                tracing::warn!(%error, start = format_args!("{start:#x}"), len = stop - start, "title memory refused by the host GPU");
                None
            }
        }
    });
    let Some(window) = window else {
        static COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let count = COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        if count <= 16 || count.is_power_of_two() {
            tracing::info!(
                count,
                address = format_args!("{address:#x}"),
                len = end - address,
                "title memory not usable in place: copied"
            );
            if count <= 3 {
                use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
                let (mut at, mut text) = (address, String::new());
                while at < end && text.len() < 1500 {
                    let mut i = MEMORY_BASIC_INFORMATION::default();
                    // SAFETY: a query of our own address space.
                    if unsafe { VirtualQuery(at as *const _, &mut i, size_of_val(&i)) } == 0 {
                        break;
                    }
                    text.push_str(&format!(
                        " {:x}+{:x}@{:x}:{:x}/{:x}/{:x}",
                        i.BaseAddress as usize, i.RegionSize, i.AllocationBase as usize, i.State, i.Type, i.Protect
                    ));
                    at = i.BaseAddress as u64 + i.RegionSize as u64;
                }
                tracing::info!("its regions:{text}");
            }
        }
        refused.insert(address, (end, frame));
        return None;
    };
    let count = map.len() + 1;
    if count <= 16 || count.is_power_of_two() {
        tracing::info!(
            count,
            start = format_args!("{:#x}", window.start),
            len = window.len,
            "title memory in place"
        );
    }
    let window = Arc::new(window);
    map.insert(window.start, window.clone());
    Some(window)
}

/// The title mapped or unmapped memory in `[start, end)`.
pub(crate) fn remapped(start: u64, end: u64) {
    use windows::core::Interface;
    let mut map = windows_of_memory();
    if map.is_empty() {
        return;
    }
    let over: Vec<u64> = map
        .range(..end)
        .filter(|(_, s)| s.start + s.len > start)
        .map(|(k, _)| *k)
        .collect();
    for key in over {
        let Some(s) = map.remove(&key) else { continue };
        let mut gone = GONE.lock().unwrap_or_else(|e| e.into_inner());
        gone.insert(s.buffer.as_raw() as usize);
        if let Some(d) = &s.direct {
            d.dead.store(true, Ordering::Release);
            for (_, a) in d.aliases.lock().unwrap_or_else(|e| e.into_inner()).iter() {
                gone.insert(a.buffer.as_raw() as usize);
                if let Some(d) = &a.direct {
                    d.dead.store(true, Ordering::Release);
                }
            }
        }
        // shortcut: kept for good, pages too (`GONE` names its buffers by
        // address); drop them with their last view if the title remaps much.
        std::mem::forget(s);
        static COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let count = COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        if count.is_power_of_two() {
            tracing::info!(count, "title memory in place was mapped anew");
        }
    }
}

/// The memory behind a buffer view with elements of `element` bytes, of
/// `len` bytes at `address`.
pub(crate) fn for_view(address: u64, len: u64, element: u64) -> Option<Arc<Shadow>> {
    let shadow = find(address, len)?;
    if shadow.sparse.is_some() {
        return shadow.sparse_on_element(address, element);
    }
    if !shadow.is_direct() {
        return Some(shadow);
    }
    shadow.on_element(address, element).or_else(|| {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(address = format_args!("{address:#x}"), element, "a view off its elements: copied");
        });
        let copy = find_copy(address, len)?;
        if copy.sparse.is_some() {
            return copy.sparse_on_element(address, element);
        }
        Some(copy)
    })
}

/// Copies of title memory the host GPU cannot use in place: a buffer as long
/// as the address space chunk it stands for, with memory behind the 64 KiB
/// tiles that were brought up to date. Reads of the others give zeros, where
/// a copy ending too soon was a GPU page fault; and memory many buffers of
/// the title overlap in is copied once.
/// A quick hash for tile numbers.
#[derive(Clone, Copy, Default)]
struct Mix(u64);
impl std::hash::BuildHasher for Mix {
    type Hasher = Mix;
    fn build_hasher(&self) -> Mix {
        Mix(0)
    }
}
impl std::hash::Hasher for Mix {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u32(u32::from(b));
        }
    }
    fn write_u32(&mut self, x: u32) {
        self.0 = (self.0.rotate_left(5) ^ u64::from(x)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

struct Sparse {
    /// By tile: where the CPU writes it, and where it is in which heap.
    tiles: Mutex<std::collections::HashMap<u32, Tile, Mix>>,
    /// What tiles shaders write held before the submission.
    before: Mutex<std::collections::HashMap<u32, Vec<u8>, Mix>>,
    /// By tile: the frame to look at it next in, and how often in a row it
    /// had not changed.
    looks: Mutex<std::collections::HashMap<u32, (u64, u32), Mix>>,
    /// The same from other 64 KiB starts ([`Direct::aliases`]).
    aliases: Mutex<Vec<Arc<Shadow>>>,
    /// This is an alias of that one.
    of: Option<Arc<Shadow>>,
}

#[derive(Clone, Copy)]
struct Tile {
    data: usize,
    heap: usize,
    offset: u32,
}

/// Address space a sparse copy stands for.
const CHUNK: u64 = 2 << 30;

/// Memory for tiles, in heaps the CPU maps.
struct TilePool {
    heaps: Vec<(windows::Win32::Graphics::Direct3D12::ID3D12Heap, ID3D12Resource, usize)>,
    /// Tiles used of the last one.
    used: u32,
}
// SAFETY: free-threaded objects and a persistent mapping.
unsafe impl Send for TilePool {}

static TILES: Mutex<TilePool> = Mutex::new(TilePool {
    heaps: Vec::new(),
    used: 0,
});

const POOL_TILES: u32 = 512;

fn new_tile() -> Option<Tile> {
    use windows::Win32::Graphics::Direct3D12::*;
    let mut pool = TILES.lock().unwrap_or_else(|e| e.into_inner());
    if pool.heaps.is_empty() || pool.used == POOL_TILES {
        let size = u64::from(POOL_TILES) * PAGE as u64;
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: size,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
        };
        let mut heap: Option<ID3D12Heap> = None;
        let mut whole: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation; the mapping lives as long as the heap.
        let made = unsafe {
            let device = &host()?.device;
            device.CreateHeap(&desc, &mut heap).and_then(|()| {
                device.CreatePlacedResource(
                    heap.as_ref(),
                    0,
                    &buffer_desc(size, D3D12_RESOURCE_FLAG_NONE),
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut whole,
                )
            })
            .and_then(|()| match &whole {
                Some(w) => w.Map(0, None, Some(&mut data)),
                None => Err(windows::core::Error::empty()),
            })
        };
        if let Err(error) = made {
            tracing::warn!(%error, "no memory for copies of title memory");
            return None;
        }
        pool.heaps.push((heap?, whole?, data as usize));
        pool.used = 0;
    }
    let heap = pool.heaps.len() - 1;
    let offset = pool.used;
    pool.used += 1;
    crate::memstat::add(&crate::memstat::SHADOWS, PAGE as u64);
    Some(Tile {
        data: pool.heaps[heap].2 + offset as usize * PAGE,
        heap,
        offset,
    })
}

fn buffer_desc(
    len: u64,
    flags: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_FLAGS,
) -> windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC {
    use windows::Win32::Graphics::Direct3D12::*;
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: len,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

/// Sparse copies by chunk.
static SPARSE: Mutex<BTreeMap<u64, Option<Arc<Shadow>>>> = Mutex::new(BTreeMap::new());

fn sparse_buffer(start: u64, len: u64, of: Option<Arc<Shadow>>) -> Option<Shadow> {
    use windows::Win32::Graphics::Direct3D12::*;
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation of a buffer without memory.
    let made = unsafe {
        host()?.device.CreateReservedResource(
            &buffer_desc(len, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        )
    };
    if let Err(error) = made {
        tracing::warn!(%error, len, "no sparse buffer");
        return None;
    }
    let buffer = buffer?;
    crate::host::name(&buffer, || format!("copies of title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::info!(start = format_args!("{start:#x}"), len, alias = of.is_some(), "sparse copies of title memory");
    Some(Shadow {
        buffer,
        data: std::ptr::null_mut(),
        gpu,
        start,
        len,
        copied: Mutex::new(if of.is_some() { Vec::new() } else { vec![0; (len / PAGE as u64) as usize] }),
        watched: AtomicU8::new(0),
        pending: Mutex::default(),
        before: Mutex::default(),
        direct: None,
        unwatched: Mutex::default(),
        sparse: Some(Sparse {
            tiles: Mutex::default(),
            before: Mutex::default(),
            looks: Mutex::default(),
            aliases: Mutex::default(),
            of,
        }),
    })
}

/// The sparse copy `[address, end)` lies in.
fn sparse(address: u64, end: u64) -> Option<Arc<Shadow>> {
    if cfg!(test) || (end - 1) / CHUNK != address / CHUNK {
        return None;
    }
    SPARSE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .entry(address / CHUNK)
        .or_insert_with(|| sparse_buffer(address / CHUNK * CHUNK, CHUNK, None).map(Arc::new))
        .clone()
}

/// Maps tiles of a sparse buffer (`first`, then each the next) to heaps.
fn map_tiles(buffer: &ID3D12Resource, tiles: &[(u32, Tile)]) {
    use windows::Win32::Graphics::Direct3D12::*;
    let Some(host) = host() else { return };
    let pool = TILES.lock().unwrap_or_else(|e| e.into_inner());
    for &(at, tile) in tiles {
        let coordinate = D3D12_TILED_RESOURCE_COORDINATE {
            X: at,
            ..Default::default()
        };
        let region = D3D12_TILE_REGION_SIZE {
            NumTiles: 1,
            ..Default::default()
        };
        // SAFETY: a live sparse buffer and heap, tiles inside both.
        unsafe {
            host.queue().UpdateTileMappings(
                buffer,
                1,
                Some(&coordinate),
                Some(&region),
                &pool.heaps[tile.heap].0,
                1,
                Some(&D3D12_TILE_RANGE_FLAG_NONE),
                Some(&tile.offset),
                Some(&1),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
    }
}

impl Shadow {
    /// The copy whose tiles these are.
    fn sparse_root(&self) -> Option<(&Shadow, &Sparse)> {
        let s = self.sparse.as_ref()?;
        match &s.of {
            Some(root) => Some((root, root.sparse.as_ref()?)),
            None => Some((self, s)),
        }
    }

    /// The CPU's view of the tile `address` is in, with memory behind it.
    fn tile(&self, s: &Sparse, address: usize) -> Option<usize> {
        let at = ((address - self.start as usize) / PAGE) as u32;
        let mut tiles = s.tiles.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(t) = tiles.get(&at) {
            return Some(t.data);
        }
        let tile = new_tile()?;
        tiles.insert(at, tile);
        map_tiles(&self.buffer, &[(at, tile)]);
        for alias in s.aliases.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            let shift = ((alias.start - self.start) / PAGE as u64) as u32;
            if at >= shift {
                map_tiles(&alias.buffer, &[(at - shift, tile)]);
            }
        }
        Some(tile.data)
    }

    /// Runs `f` on the CPU's view of each tile's part of `[from, to)`: its
    /// title address, the view, its length.
    fn each_tile(&self, s: &Sparse, from: usize, to: usize, mut f: impl FnMut(usize, *mut u8, usize)) {
        let mut at = from;
        while at < to {
            let n = (PAGE - at % PAGE).min(to - at);
            if let Some(data) = self.tile(s, at) {
                f(at, (data + at % PAGE) as *mut u8, n);
            }
            at += n;
        }
    }

    /// A sparse copy from a start `address` is a multiple of `element` bytes
    /// after.
    fn sparse_on_element(self: &Arc<Self>, address: u64, element: u64) -> Option<Arc<Shadow>> {
        if element == 0 || (address - self.start) % element == 0 {
            return Some(self.clone());
        }
        let s = self.sparse.as_ref()?;
        let page = PAGE as u64;
        let mut base = self.start + page;
        while base <= address && base - self.start <= 4096 * page {
            if (address - base) % element == 0 {
                let mut aliases = s.aliases.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(a) = aliases.iter().find(|a| a.start == base) {
                    return Some(a.clone());
                }
                let len = self.start + self.len - base;
                let made = Arc::new(sparse_buffer(base, len, Some(self.clone()))?);
                aliases.push(made.clone());
                drop(aliases);
                let shift = ((base - self.start) / page) as u32;
                let known: Vec<(u32, Tile)> = s
                    .tiles
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .iter()
                    .filter(|(at, _)| **at >= shift)
                    .map(|(at, t)| (at - shift, *t))
                    .collect();
                map_tiles(&made.buffer, &known);
                return Some(made);
            }
            base += page;
        }
        None
    }

    /// [`Shadow::writing`] of a sparse copy.
    fn sparse_writing(&self, s: &Sparse, address: usize, len: usize) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address + len));
        let mut before = s.before.lock().unwrap_or_else(|e| e.into_inner());
        let start = self.start as usize;
        self.each_tile(s, address, address + len, |at, data, _| {
            let tile = ((at - start) / PAGE) as u32;
            before.entry(tile).or_insert_with(|| {
                let whole = data as usize & !(PAGE - 1);
                // SAFETY: the tile's mapping.
                unsafe { std::slice::from_raw_parts(whole as *const u8, PAGE) }.to_vec()
            });
        });
    }

    /// [`Shadow::copy_back`] of a sparse copy.
    fn sparse_back(&self, s: &Sparse, address: usize, len: usize) -> usize {
        let mut before = s.before.lock().unwrap_or_else(|e| e.into_inner());
        let start = self.start as usize;
        let mut written = 0;
        let mut title = vec![0u8; PAGE];
        self.each_tile(s, address, address + len, |at, data, n| {
            let Some(old) = before.get_mut(&(((at - start) / PAGE) as u32)) else {
                return;
            };
            let old = &mut old[at % PAGE..at % PAGE + n];
            // SAFETY: the tile's mapping; only this thread writes it.
            let host = unsafe { std::slice::from_raw_parts_mut(data, n) };
            if host == old || !crate::host::read_into(at as *const u8, &mut title[..n]) {
                return;
            }
            let mut changed = false;
            for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old.iter_mut()) {
                if *h != *o && *t != *h {
                    *t = *h;
                    changed = true;
                }
                *h = *t;
                *o = *t;
            }
            if changed {
                crate::watch::writing(at, n);
                crate::host::write_mapped(at as *mut u8, title.as_ptr(), n);
                written += n;
            }
        });
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&r| r != (address, address + len));
        if pending.is_empty() {
            before.clear();
        }
        written
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    let gone = buffers().remove(&(start, resource));
    if let Some(shadow) = gone.and_then(|e| e.shadow) {
        crate::host::retire(shadow);
    }
}

/// Whether a title buffer holds `[address, address + len)` (its memory is
/// the title's, without asking the kernel).
pub(crate) fn in_buffer(address: u64, len: u64) -> bool {
    let Some(end) = address.checked_add(len.max(1)) else {
        return false;
    };
    buffers()
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .any(|(_, e)| e.end >= end)
}

/// The copy of a title buffer holding `[address, address + len)`: of one
/// whose copy is made already if there is one (a buffer inside another,
/// say in the world's 174 MB heap, then shares its copy instead of paying
/// for its bytes twice, and the GPU's writes land in one copy), else of
/// the innermost, made now.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    if let Some(window) = direct(address, end) {
        return Some(window);
    }
    find_copy(address, len)
}

fn find_copy(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    if let Some(copy) = sparse(address, end) {
        return Some(copy);
    }
    let mut map = buffers();
    let holding: Vec<((u64, usize), bool)> = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .filter(|(_, e)| e.end >= end)
        .map(|(k, e)| (*k, e.shadow.is_some()))
        .collect();
    let key = holding
        .iter()
        .find(|(_, made)| *made)
        .or(holding.first())
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, data, gpu, key.0, len)));
    }
    entry.shadow.clone()
}

/// The copy holding `address` for a view without a size (a root view, which
/// the host does not bound): in memory outside buffers, one with at least
/// `reach` bytes after `address`. A window ending sooner than the data read
/// through the view is a GPU page fault.
pub(crate) fn find_reaching(address: u64, reach: u64) -> Option<Arc<Shadow>> {
    // To the end of the innermost buffer holding it: a window ending sooner
    // faults.
    let end = buffers()
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .map(|(_, e)| e.end)
        .find(|&e| e > address);
    if let Some(end) = end {
        return direct(address, end).or_else(|| find_copy(address, 1));
    }
    if let Some(window) = direct(address, address.checked_add(reach)?) {
        return Some(window);
    }
    loose(address, address.checked_add(reach)?)
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    if let Some(copy) = sparse(address, end) {
        return Some(copy);
    }
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    // Any window copied already that holds the range: a range crossing a
    // window's end used to get a second copy of the same bytes.
    if let Some(shadow) = map
        .range(..=(start, u64::MAX))
        .rev()
        .take(64)
        .find(|((s, l), _)| *s <= address && s + l >= end)
        .map(|(_, shadow)| shadow.clone())
    {
        return Some(shadow);
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len));
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Range;

    #[test]
    fn partial_match_leaves_unchecked_bytes_visible() {
        let mut unchanged = UnchangedPages::default();
        let before = vec![0; PAGE];
        let mut host = before.clone();
        host[PAGE - 1] = 1;
        unchanged.begin(PAGE, 2 * PAGE);

        assert!(!unchanged.differs(PAGE, &host[..4], &before[..4]));
        assert!(!unchanged.pages[0]);
        assert!(unchanged.differs(PAGE, &host, &before));
        assert!(!unchanged.pages[0]);
    }

    #[test]
    fn snapshot_edges_are_cacheable_and_survive_later_ranges() {
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(PAGE - 3, PAGE + 4);
        assert!(!unchanged.differs(PAGE - 3, &[1; 3], &[1; 3]));
        assert!(!unchanged.differs(PAGE, &[2; 4], &[2; 4]));
        assert_eq!(unchanged.pages, [true, true]);

        unchanged.begin(PAGE - 3, PAGE + 4);
        assert!(!unchanged.differs(PAGE - 2, &[1; 2], &[1; 2]));
        assert!(!unchanged.differs(PAGE + 1, &[2; 2], &[2; 2]));
        assert_eq!(unchanged.pages, [true, true]);
    }

    #[test]
    fn next_submission_checks_the_same_page_again() {
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(0, 4);
        assert!(!unchanged.differs(0, &[0; 4], &[0; 4]));
        unchanged.clear();
        unchanged.begin(0, 4);
        assert!(unchanged.differs(0, &[0, 0, 0, 1], &[0; 4]));
    }

    // Both versions walk exactly the same declared ranges, in their
    // original order. Only the page comparison differs.
    fn scan(
        host: &[u8],
        before: &[u8],
        ranges: &[Range<usize>],
        mut differs: impl FnMut(usize, &[u8], &[u8]) -> bool,
    ) -> usize {
        let mut changed = 0;
        for range in ranges {
            let mut at = range.start;
            while at < range.end {
                let n = (PAGE - at % PAGE).min(range.end - at);
                changed += usize::from(differs(at, &host[at..at + n], &before[at..at + n]));
                at += n;
            }
        }
        changed
    }

    #[test]
    fn overlapping_ranges_match_uncached_comparisons() {
        let before = vec![0; 4 * PAGE];
        let mut host = before.clone();
        for at in [2, PAGE - 1, PAGE + 5, 3 * PAGE + 7] {
            host[at] = 1;
        }
        let ranges = [
            0..4,
            2..2 * PAGE + 3,
            PAGE - 2..4 * PAGE,
            2 * PAGE..3 * PAGE,
        ];
        let uncached = scan(&host, &before, &ranges, |_, host, before| host != before);
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(0, before.len());
        let cached = scan(&host, &before, &ranges, |at, host, before| {
            unchanged.differs(at, host, before)
        });
        assert_eq!(uncached, 6);
        assert_eq!(cached, uncached);
        assert_eq!(unchanged.pages, [false, false, true, false]);
    }

    // cargo test --release -p victory-gpu benchmark_overlapping_copy_back -- --ignored --nocapture
    #[test]
    #[ignore = "synthetic CPU release benchmark"]
    fn benchmark_overlapping_copy_back() {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        const SIZE: usize = 128 << 20;
        let before = vec![0xA5; SIZE];
        let mut host = before.clone();
        for page in (0..SIZE / PAGE).step_by(64) {
            host[page * PAGE + PAGE / 2] ^= 0xFF;
        }
        let ranges: Vec<_> = (0..24).map(|i| i * PAGE + i..SIZE - i * 13).collect();
        let mut baseline = [Duration::ZERO; 5];
        let mut cached = [Duration::ZERO; 5];
        let mut unchanged = UnchangedPages::default();
        for (baseline, cached) in baseline.iter_mut().zip(&mut cached) {
            let started = Instant::now();
            let expected = scan(
                black_box(&host),
                black_box(&before),
                black_box(&ranges),
                |_, host, before| host != before,
            );
            *baseline = started.elapsed();

            unchanged.clear();
            let started = Instant::now();
            unchanged.begin(0, before.len());
            let actual = scan(
                black_box(&host),
                black_box(&before),
                black_box(&ranges),
                |at, host, before| unchanged.differs(at, host, before),
            );
            *cached = started.elapsed();
            assert_eq!(actual, expected);
        }
        baseline.sort_unstable();
        cached.sort_unstable();
        println!(
            "median of five runs, 24 overlapping ranges across 128 MiB: {:?} -> {:?} ({:.1}x)",
            baseline[2],
            cached[2],
            baseline[2].as_secs_f64() / cached[2].as_secs_f64(),
        );
    }
}
