// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared GPU request helpers.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::config::CDI_GPU_DEVICE_ALL;
use crate::proto::compute::v1::DriverGpuResourceRequirement;

const CDI_NVIDIA_GPU_PREFIX: &str = "nvidia.com/gpu=";
const CDI_NVIDIA_GPU_ALL_SUFFIX: &str = "all";

/// Normalized CDI GPU inventory used by local container drivers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CdiGpuInventory {
    device_ids: Vec<String>,
}

impl CdiGpuInventory {
    /// Build a normalized inventory from runtime-reported CDI device IDs.
    ///
    /// Non-NVIDIA GPU IDs are ignored. Duplicate IDs are removed. For default
    /// selection, indexed IDs and UUID-style IDs are treated as separate naming
    /// families because they may refer to the same devices.
    #[must_use]
    pub fn new(device_ids: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        let mut device_ids = device_ids
            .into_iter()
            .filter_map(|id| {
                let id = id.as_ref().trim();
                id.starts_with(CDI_NVIDIA_GPU_PREFIX)
                    .then(|| id.to_string())
            })
            .collect::<Vec<_>>();
        device_ids.sort();
        device_ids.dedup();
        Self { device_ids }
    }

    #[must_use]
    pub fn as_slice(&self) -> &[String] {
        &self.device_ids
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.device_ids.is_empty()
    }

    fn default_device_family(&self) -> Result<Vec<String>, CdiGpuSelectionError> {
        let mut indexed = self
            .device_ids
            .iter()
            .filter_map(|id| {
                let suffix = cdi_nvidia_gpu_suffix(id)?;
                let index = suffix.parse::<u64>().ok()?;
                Some((index, id.clone()))
            })
            .collect::<Vec<_>>();
        if !indexed.is_empty() {
            indexed.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
            return Ok(indexed.into_iter().map(|(_, id)| id).collect());
        }

        let mut uuid_style = self
            .device_ids
            .iter()
            .filter_map(|id| {
                let suffix = cdi_nvidia_gpu_suffix(id)?;
                (suffix != CDI_NVIDIA_GPU_ALL_SUFFIX).then(|| id.clone())
            })
            .collect::<Vec<_>>();
        if !uuid_style.is_empty() {
            uuid_style.sort();
            return Ok(uuid_style);
        }

        if self.device_ids.iter().any(|id| id == CDI_GPU_DEVICE_ALL) {
            return Ok(vec![CDI_GPU_DEVICE_ALL.to_string()]);
        }

        Err(CdiGpuSelectionError::NoAvailableDevices)
    }
}

/// Concurrency-safe round-robin cursor for default CDI GPU selection.
#[derive(Debug, Default)]
pub struct CdiGpuRoundRobin {
    next: AtomicUsize,
}

impl CdiGpuRoundRobin {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            next: AtomicUsize::new(0),
        }
    }

    /// Return the next default device ID and advance the cursor.
    pub fn next_default_device_id(
        &self,
        inventory: &CdiGpuInventory,
    ) -> Result<String, CdiGpuSelectionError> {
        self.next_default_device_ids(inventory, 1)
            .map(|devices| devices[0].clone())
    }

    /// Return the current default device ID without advancing the cursor.
    pub fn peek_default_device_id(
        &self,
        inventory: &CdiGpuInventory,
    ) -> Result<String, CdiGpuSelectionError> {
        self.peek_default_device_ids(inventory, 1)
            .map(|devices| devices[0].clone())
    }

    /// Return the next default device IDs and advance the cursor by `count`.
    pub fn next_default_device_ids(
        &self,
        inventory: &CdiGpuInventory,
        count: usize,
    ) -> Result<Vec<String>, CdiGpuSelectionError> {
        self.selected_default_device_ids(inventory, count, true)
    }

    /// Return the current default device IDs without advancing the cursor.
    pub fn peek_default_device_ids(
        &self,
        inventory: &CdiGpuInventory,
        count: usize,
    ) -> Result<Vec<String>, CdiGpuSelectionError> {
        self.selected_default_device_ids(inventory, count, false)
    }

    fn selected_default_device_ids(
        &self,
        inventory: &CdiGpuInventory,
        count: usize,
        consume: bool,
    ) -> Result<Vec<String>, CdiGpuSelectionError> {
        if count == 0 {
            return Err(CdiGpuSelectionError::InvalidCount);
        }
        let devices = inventory.default_device_family()?;
        if count > devices.len() {
            return Err(CdiGpuSelectionError::InsufficientAvailableDevices {
                requested: count,
                available: devices.len(),
            });
        }
        let base = if consume {
            self.next.fetch_add(count, Ordering::Relaxed)
        } else {
            self.next.load(Ordering::Relaxed)
        };
        let start = base % devices.len();
        Ok((0..count)
            .map(|offset| devices[(start + offset) % devices.len()].clone())
            .collect())
    }
}

/// CDI GPU selection failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CdiGpuSelectionError {
    NoAvailableDevices,
    InvalidCount,
    InsufficientAvailableDevices { requested: usize, available: usize },
    MissingDefaultDevice,
}

impl fmt::Display for CdiGpuSelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAvailableDevices => f.write_str("no NVIDIA CDI GPU devices were discovered"),
            Self::InvalidCount => f.write_str("GPU count must be greater than 0"),
            Self::InsufficientAvailableDevices {
                requested,
                available,
            } => write!(
                f,
                "GPU count request exceeds discovered NVIDIA CDI GPU devices ({requested} > {available})"
            ),
            Self::MissingDefaultDevice => {
                f.write_str("GPU request requires selected default CDI GPU device IDs")
            }
        }
    }
}

impl std::error::Error for CdiGpuSelectionError {}

/// Return the number of driver-selected CDI GPU IDs needed for this request.
///
/// `None` means no GPU was requested or explicit device IDs were supplied.
pub fn cdi_gpu_default_device_count(
    gpu: Option<&DriverGpuResourceRequirement>,
) -> Result<Option<usize>, CdiGpuSelectionError> {
    let Some(gpu) = gpu else {
        return Ok(None);
    };
    if !gpu.device_ids.is_empty() {
        return Ok(None);
    }
    let count =
        usize::try_from(gpu.count.unwrap_or(1)).map_err(|_| CdiGpuSelectionError::InvalidCount)?;
    if count == 0 {
        return Err(CdiGpuSelectionError::InvalidCount);
    }
    Ok(Some(count))
}

/// Resolve a driver GPU request into CDI device identifiers.
///
/// `None` means no GPU was requested. A GPU request with explicit device IDs
/// passes through unchanged. A GPU request with no explicit device IDs uses the
/// driver-selected default CDI IDs.
pub fn cdi_gpu_device_ids(
    gpu: Option<&DriverGpuResourceRequirement>,
    selected_default_devices: Option<&[String]>,
) -> Result<Option<Vec<String>>, CdiGpuSelectionError> {
    let Some(gpu) = gpu else {
        return Ok(None);
    };
    if !gpu.device_ids.is_empty() {
        return Ok(Some(gpu.device_ids.clone()));
    }
    let requested =
        cdi_gpu_default_device_count(Some(gpu))?.expect("GPU request has default count");
    let devices = selected_default_devices.ok_or(CdiGpuSelectionError::MissingDefaultDevice)?;
    if devices.len() != requested {
        return Err(CdiGpuSelectionError::MissingDefaultDevice);
    }
    Ok(Some(devices.to_vec()))
}

fn cdi_nvidia_gpu_suffix(id: &str) -> Option<&str> {
    id.strip_prefix(CDI_NVIDIA_GPU_PREFIX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_request(device_ids: Vec<&str>, count: Option<u32>) -> DriverGpuResourceRequirement {
        DriverGpuResourceRequirement {
            device_ids: device_ids.into_iter().map(str::to_string).collect(),
            count,
        }
    }

    #[test]
    fn cdi_gpu_device_ids_returns_none_when_absent() {
        assert_eq!(cdi_gpu_device_ids(None, None), Ok(None));
    }

    #[test]
    fn cdi_gpu_device_ids_uses_selected_default_device() {
        let request = gpu_request(vec![], None);
        let selected = vec!["nvidia.com/gpu=0".to_string()];

        assert_eq!(
            cdi_gpu_device_ids(Some(&request), Some(&selected)),
            Ok(Some(vec!["nvidia.com/gpu=0".to_string()]))
        );
    }

    #[test]
    fn cdi_gpu_device_ids_uses_selected_count_devices() {
        let request = gpu_request(vec![], Some(2));
        let selected = vec![
            "nvidia.com/gpu=0".to_string(),
            "nvidia.com/gpu=1".to_string(),
        ];

        assert_eq!(
            cdi_gpu_device_ids(Some(&request), Some(&selected)),
            Ok(Some(selected))
        );
    }

    #[test]
    fn cdi_gpu_device_ids_rejects_missing_default_device() {
        let request = gpu_request(vec![], None);

        assert_eq!(
            cdi_gpu_device_ids(Some(&request), None),
            Err(CdiGpuSelectionError::MissingDefaultDevice)
        );
    }

    #[test]
    fn cdi_gpu_device_ids_rejects_wrong_selected_device_count() {
        let request = gpu_request(vec![], Some(2));
        let selected = vec!["nvidia.com/gpu=0".to_string()];

        assert_eq!(
            cdi_gpu_device_ids(Some(&request), Some(&selected)),
            Err(CdiGpuSelectionError::MissingDefaultDevice)
        );
    }

    #[test]
    fn cdi_gpu_device_ids_rejects_zero_count() {
        let request = gpu_request(vec![], Some(0));

        assert_eq!(
            cdi_gpu_device_ids(Some(&request), Some(&[])),
            Err(CdiGpuSelectionError::InvalidCount)
        );
    }

    #[test]
    fn cdi_gpu_device_ids_passes_device_ids_through() {
        let request = gpu_request(vec!["nvidia.com/gpu=0", "nvidia.com/gpu=1"], None);

        assert_eq!(
            cdi_gpu_device_ids(Some(&request), None),
            Ok(Some(vec![
                "nvidia.com/gpu=0".to_string(),
                "nvidia.com/gpu=1".to_string()
            ]))
        );
    }

    #[test]
    fn default_device_count_ignores_explicit_device_ids() {
        let request = gpu_request(vec!["nvidia.com/gpu=0"], Some(2));

        assert_eq!(cdi_gpu_default_device_count(Some(&request)), Ok(None));
    }

    #[test]
    fn inventory_filters_and_deduplicates_nvidia_gpu_ids() {
        let inventory = CdiGpuInventory::new([
            "nvidia.com/gpu=1",
            "vendor.example/device=0",
            "nvidia.com/gpu=1",
            " nvidia.com/gpu=0 ",
        ]);

        assert_eq!(
            inventory.as_slice(),
            &vec![
                "nvidia.com/gpu=0".to_string(),
                "nvidia.com/gpu=1".to_string()
            ]
        );
    }

    #[test]
    fn round_robin_prefers_indexed_family_and_sorts_numerically() {
        let inventory = CdiGpuInventory::new([
            "nvidia.com/gpu=10",
            "nvidia.com/gpu=UUID-b",
            "nvidia.com/gpu=2",
            "nvidia.com/gpu=all",
        ]);
        let selector = CdiGpuRoundRobin::new();

        assert_eq!(
            selector.next_default_device_id(&inventory),
            Ok("nvidia.com/gpu=2".to_string())
        );
        assert_eq!(
            selector.next_default_device_id(&inventory),
            Ok("nvidia.com/gpu=10".to_string())
        );
        assert_eq!(
            selector.next_default_device_id(&inventory),
            Ok("nvidia.com/gpu=2".to_string())
        );
    }

    #[test]
    fn round_robin_uses_uuid_family_when_no_indexed_ids_exist() {
        let inventory = CdiGpuInventory::new(["nvidia.com/gpu=UUID-b", "nvidia.com/gpu=UUID-a"]);
        let selector = CdiGpuRoundRobin::new();

        assert_eq!(
            selector.next_default_device_id(&inventory),
            Ok("nvidia.com/gpu=UUID-a".to_string())
        );
    }

    #[test]
    fn round_robin_uses_all_only_inventory() {
        let inventory = CdiGpuInventory::new([CDI_GPU_DEVICE_ALL]);
        let selector = CdiGpuRoundRobin::new();

        assert_eq!(
            selector.next_default_device_id(&inventory),
            Ok(CDI_GPU_DEVICE_ALL.to_string())
        );
    }

    #[test]
    fn round_robin_rejects_empty_inventory() {
        let inventory = CdiGpuInventory::new(["vendor.example/device=0"]);
        let selector = CdiGpuRoundRobin::new();

        assert_eq!(
            selector.next_default_device_id(&inventory),
            Err(CdiGpuSelectionError::NoAvailableDevices)
        );
    }

    #[test]
    fn peek_does_not_advance_round_robin_cursor() {
        let inventory = CdiGpuInventory::new(["nvidia.com/gpu=0", "nvidia.com/gpu=1"]);
        let selector = CdiGpuRoundRobin::new();

        assert_eq!(
            selector.peek_default_device_id(&inventory),
            Ok("nvidia.com/gpu=0".to_string())
        );
        assert_eq!(
            selector.peek_default_device_id(&inventory),
            Ok("nvidia.com/gpu=0".to_string())
        );
        assert_eq!(
            selector.next_default_device_id(&inventory),
            Ok("nvidia.com/gpu=0".to_string())
        );
        assert_eq!(
            selector.next_default_device_id(&inventory),
            Ok("nvidia.com/gpu=1".to_string())
        );
    }

    #[test]
    fn round_robin_selects_requested_count_and_advances_by_count() {
        let inventory =
            CdiGpuInventory::new(["nvidia.com/gpu=0", "nvidia.com/gpu=1", "nvidia.com/gpu=2"]);
        let selector = CdiGpuRoundRobin::new();

        assert_eq!(
            selector.next_default_device_ids(&inventory, 2),
            Ok(vec![
                "nvidia.com/gpu=0".to_string(),
                "nvidia.com/gpu=1".to_string()
            ])
        );
        assert_eq!(
            selector.next_default_device_ids(&inventory, 2),
            Ok(vec![
                "nvidia.com/gpu=2".to_string(),
                "nvidia.com/gpu=0".to_string()
            ])
        );
    }

    #[test]
    fn round_robin_rejects_count_above_inventory_size() {
        let inventory = CdiGpuInventory::new(["nvidia.com/gpu=0"]);
        let selector = CdiGpuRoundRobin::new();

        assert_eq!(
            selector.peek_default_device_ids(&inventory, 2),
            Err(CdiGpuSelectionError::InsufficientAvailableDevices {
                requested: 2,
                available: 1
            })
        );
    }
}
