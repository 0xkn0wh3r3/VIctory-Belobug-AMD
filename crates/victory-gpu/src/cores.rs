//! Which processors our own threads run on, where a processor has two kinds
//! of cores. `cpu_sets` and `keep_on_fast_cores` are fxceless/victory-fork's
//! (`framelog.rs`, commit 0b0074e0, GPL-3.0-or-later, as this).

/// The system's processor sets of group 0: each one's id, its efficiency
/// class (higher is faster) and its core (which two of them may share).
fn cpu_sets() -> Vec<(u32, u8, u8)> {
    use windows_sys::Win32::System::SystemInformation::{GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut bytes = 0u32;
    // SAFETY: a size query, then the call into a buffer of that size.
    unsafe { GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut bytes, GetCurrentProcess(), 0) };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: as above; the buffer holds `bytes` bytes, aligned for the records.
    let ok = unsafe { GetSystemCpuSetInformation(buffer.as_mut_ptr().cast(), bytes, &mut bytes, GetCurrentProcess(), 0) };
    let mut sets = Vec::new();
    let mut at = 0usize;
    while ok != 0 && at + size_of::<SYSTEM_CPU_SET_INFORMATION>() <= bytes as usize {
        // SAFETY: a record inside the buffer; each says its own size.
        let info = unsafe { &*(buffer.as_ptr().cast::<u8>().add(at).cast::<SYSTEM_CPU_SET_INFORMATION>()) };
        if info.Size == 0 {
            break;
        }
        // SAFETY: the record's type is CpuSet (the only one there is).
        let set = unsafe { &info.Anonymous.CpuSet };
        if set.Group == 0 {
            sets.push((set.Id, set.EfficiencyClass, set.CoreIndex));
        }
        at += info.Size as usize;
    }
    sets
}

/// The last-level cache of each processor set of group 0, by set id: its
/// bytes, and the mask of the processors that share it (which tells one
/// core complex from another). None known: empty.
fn l3_of_sets() -> Vec<(u32, u32, u64)> {
    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, GetSystemCpuSetInformation, RelationCache, SYSTEM_CPU_SET_INFORMATION,
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // The largest cache of each logical processor of group 0, by its index,
    // and the processors that share it.
    let mut by_processor = [(0u32, 0u64); 64];
    let mut bytes = 0u32;
    // SAFETY: a size query, then the call into a buffer of that size.
    unsafe { GetLogicalProcessorInformationEx(RelationCache, std::ptr::null_mut(), &mut bytes) };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: as above; the buffer holds `bytes` bytes, aligned for the records.
    let ok = unsafe { GetLogicalProcessorInformationEx(RelationCache, buffer.as_mut_ptr().cast(), &mut bytes) };
    let mut at = 0usize;
    while ok != 0 && at + size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>() <= bytes as usize {
        // SAFETY: a record inside the buffer; each says its own size.
        let info = unsafe { &*(buffer.as_ptr().cast::<u8>().add(at).cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>()) };
        if info.Size == 0 {
            break;
        }
        if info.Relationship == RelationCache {
            // SAFETY: a cache record (asked for), with at least its first group mask.
            let cache = unsafe { &info.Anonymous.Cache };
            let mask = unsafe { cache.Anonymous.GroupMask };
            if mask.Group == 0 {
                for (p, last) in by_processor.iter_mut().enumerate() {
                    if mask.Mask >> p & 1 != 0 && cache.CacheSize > last.0 {
                        *last = (cache.CacheSize, mask.Mask as u64);
                    }
                }
            }
        }
        at += info.Size as usize;
    }
    // Processor indices to set ids.
    let mut bytes = 0u32;
    // SAFETY: as for `cpu_sets`.
    unsafe { GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut bytes, GetCurrentProcess(), 0) };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: as above.
    let ok = unsafe { GetSystemCpuSetInformation(buffer.as_mut_ptr().cast(), bytes, &mut bytes, GetCurrentProcess(), 0) };
    let mut sizes = Vec::new();
    let mut at = 0usize;
    while ok != 0 && at + size_of::<SYSTEM_CPU_SET_INFORMATION>() <= bytes as usize {
        // SAFETY: a record inside the buffer; each says its own size.
        let info = unsafe { &*(buffer.as_ptr().cast::<u8>().add(at).cast::<SYSTEM_CPU_SET_INFORMATION>()) };
        if info.Size == 0 {
            break;
        }
        // SAFETY: the record's type is CpuSet (the only one there is).
        let set = unsafe { &info.Anonymous.CpuSet };
        if set.Group == 0 && let Some(&(size, shared)) = by_processor.get(set.LogicalProcessorIndex as usize) {
            sizes.push((set.Id, size, shared));
        }
        at += info.Size as usize;
    }
    sizes
}

/// Whether the cores of the largest last-level cache count as the fast
/// ones where all cores are of one class (`VICTORY_CACHE_CORES=0`: not).
/// On a processor with two core complexes of which one has the stacked cache
/// (Ryzen 7900X3D, 7950X3D, 9900X3D, 9950X3D) every core reports the same
/// efficiency class, so nothing was asked for: our threads ran on either
/// complex. Their work (compares, hashes, copies of tens of megabytes a
/// frame) is the kind the larger cache serves. Nothing changes on a processor
/// whose caches are all alike, or with two kinds of cores.
fn cache_cores() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_CACHE_CORES") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CACHE_CORES").is_none_or(|v| v != "0"))
}

/// The sets of the fastest class, if there are slower ones too.
fn fast_by_class(sets: &[(u32, u8, u8)]) -> Vec<u32> {
    let best = sets.iter().map(|c| c.1).max().unwrap_or(0);
    let fast: Vec<u32> = sets.iter().filter(|c| c.1 == best).map(|c| c.0).collect();
    if fast.len() == sets.len() { Vec::new() } else { fast }
}

/// The sets on the largest last-level cache (`l3`: bytes and sharers by set
/// id), if another one is smaller and the largest serves at least eight sets.
fn fast_by_cache(sets: &[(u32, u8, u8)], l3: &[(u32, u32, u64)]) -> Vec<u32> {
    let size = |id: u32| l3.iter().find(|s| s.0 == id).map_or(0, |s| s.1);
    let sizes: Vec<u32> = sets.iter().map(|s| size(s.0)).collect();
    let (Some(&most), Some(&least)) = (sizes.iter().max(), sizes.iter().min()) else {
        return Vec::new();
    };
    if least == 0 || least == most {
        return Vec::new();
    }
    let fast: Vec<u32> = sets.iter().filter(|s| size(s.0) == most).map(|s| s.0).collect();
    if fast.len() < 8 { Vec::new() } else { fast }
}

/// Whether our threads keep to one core complex where the processor has
/// several of one kind (`VICTORY_CCD_CORES=1`; off by default while it is
/// tested). On a Ryzen of two complexes whose caches are alike (7900X, 7950X,
/// 9900X, 9950X) nothing else was asked for: the submitting thread and those
/// it shares its work with ran on either, and every lock, claim and handoff
/// between two of them could cross from one complex to the other: 180 ns a
/// way on a 7900X, against 20 ns inside one.
fn ccd_cores() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_CCD_CORES") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CCD_CORES").is_some_and(|v| v == "1"))
}

/// The sets of one last-level cache (`l3`: bytes and sharers by set id), if
/// there are several: the one shared by the most sets (of two alike, the one
/// of the lowest id), at least eight of them.
fn fast_by_domain(sets: &[(u32, u8, u8)], l3: &[(u32, u32, u64)]) -> Vec<u32> {
    let shared = |id: u32| l3.iter().find(|s| s.0 == id).map_or(0, |s| s.2);
    let mut domains: Vec<(u64, Vec<u32>)> = Vec::new();
    for s in sets {
        let mask = shared(s.0);
        if mask == 0 {
            return Vec::new();
        }
        match domains.iter_mut().find(|d| d.0 == mask) {
            Some(d) => d.1.push(s.0),
            None => domains.push((mask, vec![s.0])),
        }
    }
    if domains.len() < 2 {
        return Vec::new();
    }
    let most = domains.iter().map(|d| d.1.len()).max().unwrap_or(0);
    let best = domains.into_iter().filter(|d| d.1.len() == most).min_by_key(|d| d.1.iter().copied().min());
    best.map(|d| d.1).filter(|f| f.len() >= 8).unwrap_or_default()
}

/// What decided the fast sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum By {
    Nothing,
    Class,
    Cache,
    Complex,
}

/// The fast sets: those of the fastest class if there are slower ones, else
/// those of the largest cache ([`cache_cores`]), else those of one core
/// complex ([`ccd_cores`]). `now`: as things are at this moment (else whether
/// there could be any, for threads to ask at all).
fn fast_sets(sets: &[(u32, u8, u8)], now: bool) -> (Vec<u32>, By) {
    static L3: std::sync::OnceLock<Vec<(u32, u32, u64)>> = std::sync::OnceLock::new();
    fast_sets_with(sets, || L3.get_or_init(l3_of_sets), now, (cache_cores(), ccd_cores()))
}

/// [`fast_sets`] with the caches given, and the two switches.
fn fast_sets_with<'a>(
    sets: &[(u32, u8, u8)],
    l3: impl FnOnce() -> &'a [(u32, u32, u64)],
    now: bool,
    (cache, complex): (bool, bool),
) -> (Vec<u32>, By) {
    let by_class = fast_by_class(sets);
    if !by_class.is_empty() {
        return (by_class, By::Class);
    }
    let (cache, complex) = if now { (cache, complex) } else { (true, true) };
    if !cache && !complex {
        return (Vec::new(), By::Nothing);
    }
    let l3 = l3();
    if cache {
        let by_cache = fast_by_cache(sets, l3);
        if !by_cache.is_empty() {
            return (by_cache, By::Cache);
        }
    }
    let by_domain = if complex { fast_by_domain(sets, l3) } else { Vec::new() };
    let by = if by_domain.is_empty() { By::Nothing } else { By::Complex };
    (by_domain, by)
}

/// The sets of one core of the last among `fast` (of the best class where
/// `fast` is empty), and all the others, for [`lane_core`]; `None` with fewer
/// than six such cores. (Of `fast`, so that the submitting thread stays on the
/// cache or complex that the threads it shares its work with were given.)
fn one_core(sets: &[(u32, u8, u8)], fast: &[u32]) -> Option<(Vec<u32>, Vec<u32>)> {
    let best = sets.iter().map(|c| c.1).max()?;
    let mut cores: Vec<u8> =
        sets.iter().filter(|c| if fast.is_empty() { c.1 == best } else { fast.contains(&c.0) }).map(|c| c.2).collect();
    cores.sort_unstable();
    cores.dedup();
    let core = *cores.last().filter(|_| cores.len() >= 6)?;
    let of = |own: bool| sets.iter().filter(|c| (c.2 == core) == own).map(|c| c.0).collect();
    Some((of(true), of(false)))
}

/// Whether the submitting thread has a fast core to itself, as far as this
/// process goes (`VICTORY_LANE_CORE=1`): nothing else of it runs on the
/// core's two threads. It was the one a frame waited for (94 % of it in a
/// street), and shared its core's units with whatever ran beside it.
fn lane_core() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_LANE_CORE") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LANE_CORE").is_some_and(|v| v == "1"))
}

/// Threads that asked: a handle of each, and whether it is the submitting one.
static ASKED: std::sync::Mutex<Vec<(usize, bool)>> = std::sync::Mutex::new(Vec::new());

/// What the threads of the process, and those that asked, are told now
/// ([`lane_core`], which may have changed).
pub(crate) fn settle() {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, SetProcessDefaultCpuSets, SetThreadSelectedCpuSets};
    if stage() == 0 {
        return;
    }
    let sets = cpu_sets();
    let (fast, by) = fast_sets(&sets, true);
    let one = one_core(&sets, &fast).filter(|_| lane_core());
    let others: &[u32] = one.as_ref().map_or(&[], |o| &o.1);
    let null = std::ptr::null::<u32>();
    // SAFETY: our own process and the system's own set ids (none: all).
    unsafe {
        let at = if others.is_empty() { null } else { others.as_ptr() };
        SetProcessDefaultCpuSets(GetCurrentProcess(), at, others.len() as u32);
    }
    let helpers: Vec<u32> = match &one {
        Some((own, _)) => fast.iter().copied().filter(|id| !own.contains(id)).collect(),
        None => fast.clone(),
    };
    for &(thread, lane) in ASKED.lock().unwrap_or_else(|e| e.into_inner()).iter() {
        let sets = match &one {
            Some((own, _)) if lane => own,
            _ => &helpers,
        };
        // SAFETY: a handle kept open, and the system's own set ids.
        unsafe { SetThreadSelectedCpuSets(thread as _, sets.as_ptr(), sets.len() as u32) };
    }
    tracing::info!(on = one.is_some(), core = ?one.as_ref().map(|o| &o.0), "the submitting thread has a core of its own (VICTORY_LANE_CORE)");
    tracing::info!(
        fast = fast.len(),
        of = sets.len(),
        by = ?by,
        threads = ASKED.lock().unwrap_or_else(|e| e.into_inner()).len(),
        "our threads' processors (VICTORY_CACHE_CORES, VICTORY_CCD_CORES)"
    );
}

/// `VICTORY_FAST_CORES`: 0: no thread asks; 1: the threads a frame waits for
/// one by one; 2 (the default): the threads they share their work with too;
/// 3: those that work beside the rest as well (lists translated, descriptors
/// made): more threads by then than the fast cores run at a time, while the
/// slow ones stood idle.
fn stage() -> u32 {
    static STAGE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *STAGE.get_or_init(|| std::env::var("VICTORY_FAST_CORES").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(2))
}

/// Asks the system to run the calling thread, which a frame waits for, on
/// the fast cores only, where the processor has two kinds. That fork's submit
/// thread was found on a slow one in 1,529 looks of 24,021. (A selection of
/// processors is a preference the scheduler keeps unless nothing of it is
/// free.) The title's threads are not asked for.
pub(crate) fn keep_on_fast_cores(what: &str) {
    ask(what, 1);
}

/// [`keep_on_fast_cores`] for one of the threads a submission's preparing or
/// a landing shares its work with.
pub(crate) fn keep_helper_on_fast_cores(what: &str) {
    ask(what, 2);
}

/// [`keep_on_fast_cores`] for one of many threads whose work is ahead of what
/// a frame waits for.
pub(crate) fn keep_many_on_fast_cores(what: &str) {
    ask(what, 3);
}

fn ask(what: &str, from: u32) {
    use windows_sys::Win32::System::Threading::{
        GetCurrentThreadId, OpenThread, THREAD_SET_LIMITED_INFORMATION,
    };
    if stage() < from {
        return;
    }
    static FAST: std::sync::OnceLock<(Vec<u32>, usize)> = std::sync::OnceLock::new();
    let (fast, of) = FAST.get_or_init(|| {
        let sets = cpu_sets();
        // (whether there can be fast sets at all: `VICTORY_CACHE_CORES` and
        // `VICTORY_CCD_CORES` may change while the title runs, and `settle`
        // follows them)
        (fast_sets(&sets, false).0, sets.len())
    });
    // All alike (or nothing known): nothing to ask for.
    if fast.is_empty() {
        return;
    }
    // SAFETY: our own thread; the handle stays open.
    let thread = unsafe { OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, GetCurrentThreadId()) };
    if thread.is_null() {
        tracing::warn!(what, error = %std::io::Error::last_os_error(), "the thread could not be kept on the fast cores");
        return;
    }
    ASKED.lock().unwrap_or_else(|e| e.into_inner()).push((thread as usize, what == "victory-submit"));
    settle();
    tracing::info!(what, fast = fast.len(), of, "the thread is kept on the fast cores (VICTORY_FAST_CORES=0: not)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fast_cores_are_those_of_the_best_class_if_there_is_a_worse_one() {
        assert_eq!(fast_by_class(&[(256, 1, 0), (257, 1, 0), (258, 0, 2)]), [256, 257]);
        assert!(fast_by_class(&[(256, 0, 0), (257, 0, 1)]).is_empty());
        assert!(one_core(&[(256, 1, 0), (257, 1, 0), (258, 0, 2)], &[256, 257]).is_none());
        let eight: Vec<(u32, u8, u8)> = (0..16).map(|n| (n, 1, n as u8 / 2 * 2)).chain((16..24).map(|n| (n, 0, n as u8))).collect();
        let (own, others) = one_core(&eight, &fast_by_class(&eight)).unwrap();
        assert_eq!((own, others.len()), (vec![14, 15], 22));
        assert!(fast_by_class(&[]).is_empty());
        assert!(!cpu_sets().is_empty());
        std::thread::spawn(|| keep_on_fast_cores("test")).join().unwrap();
    }

    const MB: u32 = 1 << 20;

    /// `n` sets of one class, two to a core, from id 256.
    fn alike(n: u32) -> Vec<(u32, u8, u8)> {
        (0..n).map(|i| (256 + i, 0, (i / 2) as u8)).collect()
    }

    /// The caches of `n` sets from id 256, `per` of them to a complex, with
    /// the bytes `size` gives each set.
    fn caches(n: u32, per: u32, size: impl Fn(u32) -> u32) -> Vec<(u32, u32, u64)> {
        (0..n).map(|i| (256 + i, size(i), ((1u64 << per) - 1) << (i / per * per))).collect()
    }

    #[test]
    fn the_largest_cache_is_fast_only_where_the_caches_differ() {
        // Two complexes, one with the stacked cache (9950X3D: 16 sets each).
        let sets = alike(32);
        let l3 = caches(32, 16, |i| if i < 16 { 96 * MB } else { 32 * MB });
        assert_eq!(fast_by_cache(&sets, &l3), (256..272).collect::<Vec<_>>());
        // The stacked cache on the second complex: that one.
        let flipped = caches(32, 16, |i| if i < 16 { 32 * MB } else { 96 * MB });
        assert_eq!(fast_by_cache(&sets, &flipped), (272..288).collect::<Vec<_>>());
        // Two complexes alike (9950X): nothing.
        let even = caches(32, 16, |_| 32 * MB);
        assert!(fast_by_cache(&sets, &even).is_empty());
        // One complex (7800X3D, 9800X3D): nothing.
        let one = caches(16, 16, |_| 96 * MB);
        assert!(fast_by_cache(&alike(16), &one).is_empty());
        // Caches not known: nothing.
        assert!(fast_by_cache(&sets, &[]).is_empty());
        assert!(fast_by_cache(&sets, &l3[..16]).is_empty());
        // A larger cache of under eight sets: nothing.
        let few = caches(32, 16, |i| if i < 4 { 64 * MB } else { 32 * MB });
        assert!(fast_by_cache(&sets, &few).is_empty());
    }

    #[test]
    fn two_kinds_of_cores_still_go_by_their_class() {
        // Eight fast cores (two sets each) and eight slow ones, the caches
        // told to differ too: the class decides, whatever the caches say.
        let sets: Vec<(u32, u8, u8)> =
            (0..16).map(|n| (n, 1, n as u8 / 2 * 2)).chain((16..24).map(|n| (n, 0, n as u8))).collect();
        let l3: Vec<(u32, u32, u64)> = (0..24).map(|n| (n, if n < 16 { 30 * MB } else { 4 * MB }, 1 << (n / 4))).collect();
        for now in [true, false] {
            for switches in [(true, true), (false, false)] {
                assert_eq!(fast_sets_with(&sets, || &l3, now, switches), ((0..16).collect::<Vec<_>>(), By::Class));
            }
        }
    }

    #[test]
    fn one_complex_is_kept_to_where_the_caches_are_alike() {
        // 7900X: two complexes of six cores, twelve sets each.
        let sets = alike(24);
        let l3 = caches(24, 12, |_| 32 * MB);
        let first: Vec<u32> = (256..268).collect();
        assert_eq!(fast_by_domain(&sets, &l3), first);
        assert_eq!(fast_sets_with(&sets, || &l3, true, (true, true)), (first.clone(), By::Complex));
        // Off: nothing; but threads still ask, for when it is turned on.
        assert_eq!(fast_sets_with(&sets, || &l3, true, (true, false)), (Vec::new(), By::Nothing));
        assert_eq!(fast_sets_with(&sets, || &l3, false, (true, false)).0, first);
        // The submitting thread's own core is one of that complex.
        let (own, others) = one_core(&sets, &first).unwrap();
        assert_eq!((own, others.len()), (vec![266, 267], 22));
        // One complex (7700X, 9800X3D), or complexes under eight sets (a
        // 3900X's of three cores): nothing.
        assert!(fast_by_domain(&alike(16), &caches(16, 16, |_| 32 * MB)).is_empty());
        assert!(fast_by_domain(&alike(24), &caches(24, 6, |_| 16 * MB)).is_empty());
        // Caches not known for every set: nothing.
        assert!(fast_by_domain(&sets, &l3[..12]).is_empty());
        // Of two complexes of unlike size, the larger.
        let mut uneven = caches(20, 12, |_| 32 * MB);
        uneven.iter_mut().skip(12).for_each(|c| c.2 = 1 << 40);
        assert_eq!(fast_by_domain(&alike(20), &uneven), first);
    }

    /// This machine's caches are known for every set, and one complex of
    /// them, if any is chosen, is of them.
    #[test]
    fn the_caches_here_are_known() {
        let sets = cpu_sets();
        let l3 = l3_of_sets();
        assert!(sets.iter().all(|s| l3.iter().any(|c| c.0 == s.0 && c.1 > 0 && c.2 != 0)));
        let one = fast_by_domain(&sets, &l3);
        assert!(one.iter().all(|id| sets.iter().any(|s| s.0 == *id)));
        let complexes = l3.iter().map(|c| c.2).collect::<std::collections::BTreeSet<_>>().len();
        println!("{} sets, {complexes} last-level caches, one complex: {} sets", sets.len(), one.len());
    }

    #[test]
    fn a_larger_cache_comes_before_one_complex() {
        // 9950X3D with both switches on: the stacked cache, as before.
        let sets = alike(32);
        let l3 = caches(32, 16, |i| if i < 16 { 32 * MB } else { 96 * MB });
        let stacked: Vec<u32> = (272..288).collect();
        assert_eq!(fast_sets_with(&sets, || &l3, true, (true, true)), (stacked.clone(), By::Cache));
        // The submitting thread's core is on it too (it used to be the last
        // core of the class, whatever its cache).
        let flipped = caches(32, 16, |i| if i < 16 { 96 * MB } else { 32 * MB });
        let (fast, _) = fast_sets_with(&sets, || &flipped, true, (true, false));
        let (own, _) = one_core(&sets, &fast).unwrap();
        assert_eq!(own, [270, 271]);
        // The cache switch off: one complex all the same.
        assert_eq!(fast_sets_with(&sets, || &l3, true, (false, true)), ((256..272).collect(), By::Complex));
    }
}
