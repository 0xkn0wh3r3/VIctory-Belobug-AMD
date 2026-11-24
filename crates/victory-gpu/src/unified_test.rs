//! Whether the host can make heaps of existing memory.
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;
use windows_sys::Win32::System::Memory::*;

fn buffer(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

#[test]
fn existing_memory() {
    let host = crate::host::host().unwrap();
    let device: ID3D12Device3 = host.device.cast().unwrap();
    const N: usize = 4 << 20;
    unsafe {
        let private = VirtualAlloc(std::ptr::null(), N, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        let watched = VirtualAlloc(std::ptr::null(), N, MEM_RESERVE | MEM_COMMIT | 0x20_0000, PAGE_READWRITE);
        let partial = VirtualAlloc(std::ptr::null(), N, MEM_RESERVE, PAGE_NOACCESS);
        VirtualAlloc(partial, N / 4, MEM_COMMIT, PAGE_READWRITE);
        let section = |kind| CreateFileMappingW(windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE | kind, 0, N as u32, std::ptr::null());
        let committed = section(SEC_COMMIT);
        let reserved = section(SEC_RESERVE);
        let view = MapViewOfFile(committed, FILE_MAP_ALL_ACCESS, 0, 0, N).Value;
        let rview = MapViewOfFile(reserved, FILE_MAP_ALL_ACCESS, 0, 0, N).Value;
        VirtualAlloc(rview, N / 2, MEM_COMMIT, PAGE_READWRITE);
        let mut heaps: Vec<(&str, Option<ID3D12Heap>, *mut u8)> = Vec::new();
        for (name, at) in [("private", private), ("watched", watched), ("partial", partial), ("view", view), ("reserved view", rview)] {
            let r = device.OpenExistingHeapFromAddress::<ID3D12Heap>(at);
            println!("{name}: address {:?}", r.as_ref().err());
            heaps.push((name, r.ok(), at.cast()));
        }
        for (name, handle, at) in [("section", committed, view), ("reserved section", reserved, rview)] {
            let r = device.OpenExistingHeapFromFileMapping::<ID3D12Heap>(windows::Win32::Foundation::HANDLE(handle));
            println!("{name}: file mapping {:?}", r.as_ref().err());
            heaps.push((name, r.ok(), at.cast()));
        }
        for (name, heap, at) in &heaps {
            let Some(heap) = heap else { continue };
            let d = heap.GetDesc();
            println!("{name}: size {:#x} flags {:#x} type {} page {} pool {}", d.SizeInBytes, d.Flags.0, d.Properties.Type.0, d.Properties.CPUPageProperty.0, d.Properties.MemoryPoolPreference.0);
            for (what, flags) in [
                ("plain", D3D12_RESOURCE_FLAG_NONE),
                ("uav", D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                ("cross", D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER),
                ("cross uav", D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            ] {
                let mut r: Option<ID3D12Resource> = None;
                let made = host.device.CreatePlacedResource(heap, 0, &buffer(N as u64, flags), D3D12_RESOURCE_STATE_COMMON, None, &mut r);
                println!("  placed {what}: {made:?}");
                let Some(r) = r else { continue };
                if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 { continue; }
                // The GPU writes it, the CPU sees it.
                let list = host.new_list().unwrap();
                let src = host.upload_buffer(0x1_0000);
                if let Some((s, p)) = src {
                    p.write_bytes(0xab, 0x1_0000);
                    list.list.CopyBufferRegion(&r, 0x1000, &s, 0, 0x1000);
                    list.close();
                    host.execute(&[&list]);
                    println!("  read back {:#x}", at.add(0x1800).read());
                }
            }
            // A reserved buffer over it.
            let mut res: Option<ID3D12Resource> = None;
            let made = host.device.CreateReservedResource(&buffer(64 << 20, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_COMMON, None, &mut res);
            println!("  tiled uav: {made:?}");
            let mut res2: Option<ID3D12Resource> = None;
            let made = host.device.CreateReservedResource(&buffer(64 << 20, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER), D3D12_RESOURCE_STATE_COMMON, None, &mut res2);
            println!("  tiled uav cross: {made:?}");
        }
        println!("removed: {:?}", host.device.GetDeviceRemovedReason());
    }
}
