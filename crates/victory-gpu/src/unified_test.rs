//! Whether the host can make heaps of ranges of title memory.
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;
use windows_sys::Win32::System::Memory::*;
use windows_sys::Win32::System::Threading::GetCurrentProcess;

fn buffer(size: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER,
        ..Default::default()
    }
}

#[test]
#[ignore = "host capability probe"]
fn ranges_of_title_memory() {
    let host = crate::host::host().unwrap();
    let device = host.device.cast::<ID3D12Device13>();
    println!("device13 {:?}", device.as_ref().err());
    let Ok(device) = device else { return };
    const M: usize = 1 << 20;
    unsafe {
        // Private, committed in the middle.
        let part = VirtualAlloc(std::ptr::null(), 16 * M, MEM_RESERVE | 0x20_0000, PAGE_NOACCESS).cast::<u8>();
        VirtualAlloc(part.add(4 * M).cast(), 2 * M, MEM_COMMIT, PAGE_READWRITE);
        VirtualAlloc(part.add(6 * M).cast(), 2 * M, MEM_COMMIT, PAGE_READWRITE);
        // Views of scattered pages side by side.
        let place = VirtualAlloc2(GetCurrentProcess(), std::ptr::null(), 8 * M, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, std::ptr::null_mut(), 0).cast::<u8>();
        let section = CreateFileMappingW(windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE | SEC_RESERVE, 2, 0, std::ptr::null());
        for (k, off) in [(0usize, 5u64), (1, 3), (2, 900), (3, 4000)] {
            let at = place.add(k * M);
            VirtualFree(at.cast(), M, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
            let v = MapViewOfFile3(section, GetCurrentProcess(), at.cast(), off << 20, M, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, std::ptr::null_mut(), 0);
            VirtualAlloc(at.cast(), M, MEM_COMMIT, PAGE_READWRITE);
            println!("view {k} {:?}", v.Value as usize == at as usize);
        }
        let rust = vec![1u8; 4 * M];
        let aligned = (rust.as_ptr() as usize).next_multiple_of(65536) as *mut u8;
        for (name, at, len) in [
            ("private whole committed", part.add(4 * M), 4 * M),
            ("private inner", part.add(5 * M), 2 * M),
            ("one view", place, M),
            ("views", place, 4 * M),
            ("views inner", place.add(M / 2), 2 * M),
            ("host heap", aligned, 2 * M),
        ] {
            let heap = device.OpenExistingHeapFromAddress1::<ID3D12Heap>(at.cast(), len);
            println!("{name}: {:?}", heap.as_ref().err().map(|e| e.code()));
            let Ok(heap) = heap else { continue };
            println!("  size {:#x}", heap.GetDesc().SizeInBytes);
            let mut r: Option<ID3D12Resource> = None;
            let made = host.device.CreatePlacedResource(&heap, 0, &buffer(len as u64), D3D12_RESOURCE_STATE_COMMON, None, &mut r);
            println!("  placed {:?}", made.err());
            let Some(r) = r else { continue };
            let list = host.new_list().unwrap();
            let (s, p) = host.upload_buffer(0x1000).unwrap();
            p.write_bytes(0xcd, 0x1000);
            let to = len - 0x3000;
            list.list.CopyBufferRegion(&r, to as u64, &s, 0, 0x1000);
            list.close();
            host.execute(&[&list]);
            println!("  read {:#x} removed {:?}", at.add(to + 8).read(), host.device.GetDeviceRemovedReason().err());
        }
    }
}
