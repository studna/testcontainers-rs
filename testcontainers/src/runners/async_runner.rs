use std::{collections::HashMap, time::Duration};

use async_trait::async_trait;
use bollard::{
    models::{
        ContainerCreateBody, HostConfig, HostConfigCgroupnsModeEnum, PortBinding, ResourcesUlimits,
    },
    query_parameters::{CreateContainerOptions, CreateContainerOptionsBuilder},
};

#[cfg(feature = "host-port-exposure")]
use crate::core::containers::host::HostPortExposure;
use crate::{
    core::{
        client::{Client, ClientError},
        copy::CopyToContainer,
        error::{Result, WaitContainerError},
        mounts::{AccessMode, Mount, MountType},
        network::Network,
        CgroupnsMode, ContainerState,
    },
    ContainerAsync, ContainerRequest, Image,
};

const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(60);
#[cfg(feature = "reusable-containers")]
static TESTCONTAINERS_SESSION_ID: std::sync::LazyLock<ferroid::id::ULID> =
    std::sync::LazyLock::new(ferroid::id::ULID::now);

#[doc(hidden)]
/// A unique identifier for the currently "active" `testcontainers` "session".
///
/// This identifier is used to ensure that the current "session" does not confuse
/// containers it creates with those created by previous runs of a test suite.
///
/// For reference: without a unique-per-session identifier, containers created by
/// previous test sessions that were marked `reuse`, (or where the test suite was
/// run with the `TESTCONTAINERS_COMMAND` environment variable set to `keep`), that
/// *haven't* been manually cleaned up could be incorrectly returned from methods
/// like [`Client::get_container`](Client::get_container),
/// as the container name, labels, and network would all still match.
#[cfg(feature = "reusable-containers")]
pub(crate) fn session_id() -> &'static ferroid::id::ULID {
    &TESTCONTAINERS_SESSION_ID
}

#[async_trait]
/// Helper trait to start containers asynchronously.
///
/// ## Example
///
/// ```rust,no_run
/// use testcontainers::{core::{WaitFor, IntoContainerPort}, runners::AsyncRunner, GenericImage};
///
/// async fn test_redis() {
///     let container = GenericImage::new("redis", "7.2.4")
///         .with_exposed_port(6379.tcp())
///         .with_wait_for(WaitFor::message_on_stdout("Ready to accept connections"))
///         .start()
///         .await;
/// }
/// ```
pub trait AsyncRunner<I: Image> {
    /// Starts the container and returns an instance of `ContainerAsync`.
    async fn start(self) -> Result<ContainerAsync<I>>;

    /// Pulls the image from the registry.
    /// Useful if you want to pull the image before starting the container.
    async fn pull_image(self) -> Result<ContainerRequest<I>>;
}

#[async_trait]
impl<T, I> AsyncRunner<I> for T
where
    T: Into<ContainerRequest<I>> + Send,
    I: Image,
{
    async fn start(self) -> Result<ContainerAsync<I>> {
        let mut state = StartupState::new(self.into());
        let container_id = state.create(false).await?;
        if let Some(container) = state.owner.take() {
            return Ok(container);
        }
        let container_req = state.request.take().expect("legacy request");
        let client = state.client.take().expect("initialized client");
        let network = state.network.take();
        #[cfg(feature = "host-port-exposure")]
        let host_port_exposure = state.exposure.take();

        let copy_to_sources: Vec<&CopyToContainer> = container_req.copy_to_sources().collect();

        for copy_to_source in copy_to_sources {
            client
                .copy_to_container(&container_id, copy_to_source)
                .await?;
        }

        #[cfg(feature = "watchdog")]
        if client.config.command() == crate::core::env::Command::Remove {
            crate::watchdog::register(container_id.clone());
        }

        let startup_timeout = container_req
            .startup_timeout()
            .unwrap_or(DEFAULT_STARTUP_TIMEOUT);

        tokio::time::timeout(startup_timeout, async {
            client.start_container(&container_id).await?;

            let container = ContainerAsync::new(
                container_id,
                client.clone(),
                container_req,
                network,
                #[cfg(feature = "host-port-exposure")]
                host_port_exposure,
            )
            .await?;

            let state = ContainerState::from_container(&container).await?;
            for cmd in container.image().exec_after_start(state)? {
                container.exec(cmd).await?;
            }

            Ok(container)
        })
        .await
        .map_err(|_| WaitContainerError::StartupTimeout)?
    }

    async fn pull_image(self) -> Result<ContainerRequest<I>> {
        let container_req = self.into();
        let client = Client::lazy_client().await?;
        client
            .pull_image(
                &container_req.descriptor(),
                container_req.platform().clone(),
            )
            .await?;

        Ok(container_req)
    }
}

const OWNER_LABEL: &str = "org.testcontainers.startup-owner";

/// Shared custody of one startup. Register this synchronously before spawning `run`.
/// Retain and join that original task on its original runtime; never abort `run`.
/// Join startup and close all dependent pools before calling `take_ready` or cleanup.
pub struct StartupOwner<I: Image>(
    std::sync::Arc<tokio::sync::Mutex<StartupState<I>>>,
    std::sync::Arc<std::sync::Mutex<StartupSnapshot>>,
);

/// Immutable diagnostic copy; obtaining it never acquires the startup mutex.
/// A pending phase is not proof of success, join collection or physical absence.
#[derive(Clone, Debug, Default)]
pub struct StartupSnapshot {
    pub phase: &'static str,
    pub name: String,
    pub owner_token: String,
    pub id: Option<String>,
    pub owner_present: bool,
    pub submitted: bool,
    pub collision: bool,
    pub deadline: Option<tokio::time::Instant>,
    pub failure: Option<String>,
}

impl<I: Image> Clone for StartupOwner<I> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), self.1.clone())
    }
}

impl<I: Image> std::fmt::Debug for StartupOwner<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("StartupOwner")
            .field(&self.snapshot())
            .finish()
    }
}

struct StartupState<I: Image> {
    observed: std::sync::Arc<std::sync::Mutex<StartupSnapshot>>,
    request: Option<ContainerRequest<I>>,
    client: Option<std::sync::Arc<Client>>,
    network: Option<std::sync::Arc<Network>>,
    #[cfg(feature = "host-port-exposure")]
    exposure: Option<HostPortExposure>,
    owner: Option<ContainerAsync<I>>,
    id: Option<String>,
    name: String,
    token: String,
    submitted: bool,
    collision: bool,
    started: bool,
    phase: &'static str,
    deadline: Option<tokio::time::Instant>,
    failure: Option<std::sync::Arc<crate::TestcontainersError>>,
}

impl<I: Image> StartupOwner<I> {
    /// Reserve a fresh UUID name/token without starting any asynchronous effect.
    pub fn new(mut request: ContainerRequest<I>) -> Self {
        let token = uuid::Uuid::new_v4().to_string();
        let name = format!(
            "{}-{token}",
            request
                .container_name
                .as_deref()
                .unwrap_or("testcontainers")
        );
        request.container_name = Some(name.clone());
        request.labels.insert(OWNER_LABEL.into(), token.clone());
        let mut state = StartupState::new(request);
        state.name = name;
        state.token = token;
        state.publish();
        let observed = state.observed.clone();
        Self(
            std::sync::Arc::new(tokio::sync::Mutex::new(state)),
            observed,
        )
    }

    /// Observe progress even while the original startup or cleanup future is suspended.
    pub fn snapshot(&self) -> StartupSnapshot {
        self.1
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone()
    }

    /// Execute once. Observer expiry must stop borrowing the join, never this future.
    pub async fn run(&self) -> Result<()> {
        let mut state = self
            .0
            .try_lock()
            .map_err(|_| crate::TestcontainersError::other("startup busy"))?;
        if state.started {
            return Err(crate::TestcontainersError::other("startup already used"));
        }
        state.started = true;
        state.phase = "creating";
        state.publish();
        let result = state.start_retained().await;
        match result {
            Ok(()) => {
                state.phase = "ready";
                state.publish();
                Ok(())
            }
            Err(error) => Err(state.fail(error)),
        }
    }

    /// Transfer the same owner once, after the caller has joined successful startup.
    pub async fn take_ready(&self) -> Result<ContainerAsync<I>> {
        let mut state = self
            .0
            .try_lock()
            .map_err(|_| crate::TestcontainersError::other("startup busy"))?;
        if state.phase != "ready" || state.failure.is_some() {
            return Err(crate::TestcontainersError::other(
                "startup not successfully joined",
            ));
        }
        state.phase = "transferred";
        let owner = state
            .owner
            .take()
            .ok_or_else(|| crate::TestcontainersError::other("owner already taken"));
        state.publish();
        owner
    }

    /// Retire only confirmed absence. Submitted unknown/name404 remains pending.
    /// Retain this cleanup future across observer expiry, just as for `run`.
    pub async fn close_retained(&self) -> Result<bool> {
        let Ok(mut state) = self.0.try_lock() else {
            return Ok(false);
        };
        if state.phase != "transferred" && state.phase != "retired" {
            state.phase = "closing"; // Cancellation/panic must revoke ready transfer too.
        }
        state.publish();
        match state.close().await {
            Err(error) => Err(state.fail(error)),
            Ok(retired) => {
                state.publish();
                Ok(retired)
            }
        }
    }
}

impl<I: Image> StartupState<I> {
    fn new(request: ContainerRequest<I>) -> Self {
        Self {
            observed: Default::default(),
            request: Some(request),
            client: None,
            network: None,
            #[cfg(feature = "host-port-exposure")]
            exposure: None,
            owner: None,
            id: None,
            name: String::new(),
            token: String::new(),
            submitted: false,
            collision: false,
            started: false,
            phase: "new",
            deadline: None,
            failure: None,
        }
    }

    fn publish(&self) {
        // Formatting an arbitrary error must not hold the diagnostic mutex.
        let snapshot = StartupSnapshot {
            phase: self.phase,
            name: self.name.clone(),
            owner_token: self.token.clone(),
            id: self.id.clone(),
            owner_present: self.owner.is_some(),
            submitted: self.submitted,
            collision: self.collision,
            deadline: self.deadline,
            failure: self.failure.as_ref().map(ToString::to_string),
        };
        *self
            .observed
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = snapshot;
    }

    fn fail(&mut self, error: crate::TestcontainersError) -> crate::TestcontainersError {
        let error = self
            .failure
            .get_or_insert_with(|| std::sync::Arc::new(error))
            .clone();
        self.publish();
        crate::TestcontainersError::other(error)
    }

    fn recovered_id(&self, info: bollard::models::ContainerInspectResponse) -> Result<String> {
        let labels = info
            .config
            .and_then(|config| config.labels)
            .unwrap_or_default();
        if labels.get(OWNER_LABEL) != Some(&self.token)
            || info.name.as_deref() != Some(&format!("/{}", self.name))
        {
            return Err(crate::TestcontainersError::other(
                "reserved name/token mismatch",
            ));
        }
        info.id
            .filter(|id| !id.is_empty())
            .ok_or_else(|| crate::TestcontainersError::other("recovered container has no ID"))
    }

    fn install_owner(&mut self) {
        self.owner = Some(ContainerAsync::construct_owner(
            self.id.clone().expect("acknowledged ID"),
            self.client.as_ref().expect("submitted client").clone(),
            self.request.take().expect("retained request"),
            self.network.take(),
            #[cfg(feature = "host-port-exposure")]
            self.exposure.take(),
        ));
    }

    async fn start_retained(&mut self) -> Result<()> {
        #[cfg(feature = "reusable-containers")]
        if self.request.as_ref().expect("request").reuse() != crate::ReuseDirective::Never {
            return Err(crate::TestcontainersError::other(
                "retained startup forbids reuse",
            ));
        }
        self.create(true).await?;
        self.install_owner(); // No suspension or user callback between ACK and custody.
        self.phase = "copying";
        self.publish();
        self.owner
            .as_ref()
            .expect("registered actual owner")
            .copy_to_container()
            .await?;
        let owner = self.owner.as_mut().expect("registered actual owner");
        #[cfg(feature = "watchdog")]
        if owner.docker_client().config.command() == crate::core::env::Command::Remove {
            crate::watchdog::register(owner.id().to_string());
        }
        self.phase = "readiness";
        let timeout = owner.startup_timeout();
        let deadline = tokio::time::Instant::now() + timeout;
        self.deadline = Some(deadline);
        {
            let mut observed = self
                .observed
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            observed.phase = self.phase;
            observed.deadline = self.deadline;
        }
        let readiness = owner.start_until_ready();
        tokio::pin!(readiness);
        match tokio::time::timeout_at(deadline, &mut readiness).await {
            Ok(result) => result,
            Err(_) => {
                self.failure = Some(std::sync::Arc::new(
                    WaitContainerError::StartupTimeout.into(),
                ));
                self.observed
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .failure = Some(WaitContainerError::StartupTimeout.to_string());
                let _ = readiness.await; // Same pinned future, reactor and deadline; no restart.
                Err(WaitContainerError::StartupTimeout.into())
            }
        }
    }

    async fn close(&mut self) -> Result<bool> {
        if self.phase == "retired" {
            return Ok(true);
        }
        if self.phase == "transferred" {
            return Ok(false);
        }
        if self.owner.is_none() && self.submitted {
            let client = self.client.as_ref().expect("submitted client");
            match client.inspect(&self.name).await {
                Ok(info) => {
                    self.id = Some(self.recovered_id(info)?);
                    self.install_owner(); // Store actual recovered owner before the next await.
                    self.publish();
                }
                Err(ClientError::InspectContainer(
                    bollard::errors::Error::DockerResponseServerError {
                        status_code: 404, ..
                    },
                )) => return Ok(false),
                Err(error) => return Err(error.into()),
            }
        }
        if let Some(owner) = self.owner.as_mut() {
            owner.rm_retained().await?;
        }
        self.owner.take(); // Disarmed by exact ID404, on this original caller reactor.
        self.started = true;
        self.phase = "retired";
        Ok(true)
    }

    async fn create(&mut self, retained: bool) -> Result<String> {
        let container_req = self
            .request
            .as_mut()
            .expect("request retained until create ACK");

        #[cfg(feature = "host-port-exposure")]
        {
            self.exposure = HostPortExposure::setup(container_req).await?;
        }

        let client = Client::lazy_client().await?;
        self.client = Some(client.clone());
        if retained {
            match client.inspect(&self.name).await {
                Err(ClientError::InspectContainer(
                    bollard::errors::Error::DockerResponseServerError {
                        status_code: 404, ..
                    },
                )) => {}
                Err(error) => return Err(error.into()),
                Ok(_) => {
                    self.collision = true;
                    return Err(crate::TestcontainersError::other(
                        "reserved name already exists",
                    ));
                }
            }
        }
        let mut create_options: Option<CreateContainerOptions> = None;

        let extra_hosts: Vec<_> = container_req
            .hosts()
            .map(|(key, value)| format!("{key}:{value}"))
            .collect();

        let labels = HashMap::<String, String>::from_iter(
            container_req
                .labels()
                .iter()
                .map(|(key, value)| (key.into(), value.into()))
                .chain([
                    (
                        "org.testcontainers.managed-by".into(),
                        "testcontainers".into(),
                    ),
                    #[cfg(feature = "reusable-containers")]
                    {
                        if container_req.reuse() != crate::ReuseDirective::CurrentSession {
                            Default::default()
                        } else {
                            (
                                "org.testcontainers.session-id".to_string(),
                                session_id().to_string(),
                            )
                        }
                    },
                ])
                .filter(|(_, value): &(_, String)| !value.is_empty()),
        );

        #[cfg(feature = "reusable-containers")]
        {
            use crate::{
                core::env::ConfigurationError,
                ReuseDirective::{Always, CurrentSession},
                TestcontainersError,
            };

            if !retained && matches!(container_req.reuse(), Always | CurrentSession) {
                if labels.is_empty() && container_req.container_name().is_none() {
                    return Err(TestcontainersError::Client(ClientError::Configuration(
                        ConfigurationError::MissingContainerNameAndLabels,
                    )));
                }

                if let Some(container_info) = client
                    .get_container(
                        container_req.container_name().as_deref(),
                        container_req.network().as_deref(),
                        &labels,
                    )
                    .await?
                {
                    // Check if container is running, and start it if not
                    if !container_info.is_running {
                        client.start_container(&container_info.id).await?;
                    }

                    let network = if let Some(network) = container_req.network() {
                        Network::new(network, client.clone()).await?
                    } else {
                        None
                    };

                    let id = container_info.id;
                    self.owner = Some(ContainerAsync::construct(
                        id.clone(),
                        client,
                        self.request.take().expect("unconsumed reuse request"),
                        network,
                        #[cfg(feature = "host-port-exposure")]
                        self.exposure.take(),
                    ));
                    return Ok(id);
                }
            }
        }

        let mut config = ContainerCreateBody {
            image: Some(container_req.descriptor()),
            labels: Some(labels),
            hostname: container_req.hostname().map(|h| h.to_string()),
            host_config: Some(HostConfig {
                privileged: Some(container_req.privileged()),
                extra_hosts: Some(extra_hosts),
                cgroupns_mode: container_req.cgroupns_mode().map(|mode| mode.into()),
                userns_mode: container_req.userns_mode().map(|v| v.to_string()),
                cap_add: container_req.cap_add().cloned(),
                cap_drop: container_req.cap_drop().cloned(),
                readonly_rootfs: Some(container_req.readonly_rootfs()),
                security_opt: container_req.security_opts().cloned(),
                ..Default::default()
            }),
            working_dir: container_req.working_dir().map(|dir| dir.to_string()),
            user: container_req.user().map(|user| user.to_string()),
            healthcheck: container_req
                .health_check()
                .map(|hc| hc.clone().into_health_config()),
            open_stdin: container_req.open_stdin,
            ..Default::default()
        };

        // shared memory
        if let Some(bytes) = container_req.shm_size() {
            config.host_config = config.host_config.map(|mut host_config| {
                host_config.shm_size = Some(bytes as i64);
                host_config
            });
        }

        // create network and add it to container creation
        let network = if let Some(network) = container_req.network() {
            config.host_config = config.host_config.map(|mut host_config| {
                host_config.network_mode = Some(network.to_string());
                host_config
            });
            Network::new(network, client.clone()).await?
        } else {
            None
        };

        self.network = network;
        let mut options_builder: Option<CreateContainerOptionsBuilder> = None;

        // name of the container
        if let Some(name) = container_req.container_name() {
            let options = CreateContainerOptionsBuilder::new().name(name);

            options_builder = Some(options);
        }

        // platform of the container
        if let Some(platform) = container_req.platform() {
            let options = options_builder.unwrap_or_default();
            options_builder = Some(options.platform(platform));
        } else {
            // set platform from global platform setting if available
            let options = options_builder.unwrap_or_default();
            options_builder = Some(options.platform(client.config.platform().unwrap_or_default()));
        }

        if let Some(options) = options_builder {
            create_options = Some(options.build());
        }

        // handle environment variables
        let envs: Vec<String> = container_req
            .env_vars()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        config.env = Some(envs);

        // mounts and volumes
        let mounts: Vec<_> = container_req.mounts().map(Into::into).collect();

        if !mounts.is_empty() {
            config.host_config = config.host_config.map(|mut host_config| {
                host_config.mounts = Some(mounts);
                host_config
            });
        }

        // entrypoint
        if let Some(entrypoint) = container_req.entrypoint() {
            config.entrypoint = Some(vec![entrypoint.to_string()]);
        }

        let is_container_networked = container_req
            .network()
            .as_ref()
            .map(|network| network.starts_with("container:"))
            .unwrap_or(false);

        // expose ports
        if !is_container_networked {
            let mapped_ports = container_req
                .ports()
                .map(|ports| ports.iter().map(|p| p.container_port).collect::<Vec<_>>())
                .unwrap_or_default();

            let ports_to_expose = container_req
                .expose_ports()
                .iter()
                .copied()
                .chain(mapped_ports)
                .map(|p| format!("{p}"))
                .collect();

            // exposed ports of the image + mapped ports
            config.exposed_ports = Some(ports_to_expose);
        }

        // ports
        if container_req.ports().is_some() {
            let empty: Vec<_> = Vec::new();
            let bindings = container_req.ports().unwrap_or(&empty).iter().map(|p| {
                (
                    format!("{}", p.container_port),
                    Some(vec![PortBinding {
                        host_ip: None,
                        host_port: Some(p.host_port.to_string()),
                    }]),
                )
            });

            config.host_config = config.host_config.map(|mut host_config| {
                host_config.port_bindings = Some(bindings.collect());
                host_config
            });
        } else if !is_container_networked {
            config.host_config = config.host_config.map(|mut host_config| {
                host_config.publish_all_ports = Some(true);
                host_config.port_bindings = Some(HashMap::new());
                host_config
            });
        }

        // resource ulimits
        if let Some(ulimits) = &container_req.ulimits {
            config.host_config = config.host_config.map(|mut host_config| {
                host_config.ulimits = Some(
                    ulimits
                        .iter()
                        .map(|ulimit| ResourcesUlimits {
                            name: ulimit.name.clone(),
                            soft: ulimit.soft,
                            hard: ulimit.hard,
                        })
                        .collect(),
                );
                host_config
            });
        }

        #[cfg(feature = "device-requests")]
        {
            // device requests
            if let Some(device_requests) = &container_req.device_requests {
                config.host_config = config.host_config.map(|mut host_config| {
                    host_config.device_requests = Some(device_requests.clone());
                    host_config
                });
            }
        }

        if let Some(modifier) = container_req.host_config_modifier() {
            config.host_config = config.host_config.map(|mut host_config| {
                modifier(&mut host_config);
                host_config
            });
        }

        let cmd: Vec<_> = container_req.cmd().map(|v| v.to_string()).collect();
        if !cmd.is_empty() {
            config.cmd = Some(cmd);
        }

        let observed = self.observed.clone();
        let publish_submission = |submitted| {
            observed
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .submitted = submitted;
        };
        // The submitted bit is durable before either POST is first polled.
        self.submitted = true;
        publish_submission(true);
        let create_result = client
            .create_container(create_options.clone(), config.clone())
            .await;
        let final_result = match create_result {
            Ok(id) => Ok(id),
            Err(ClientError::CreateContainer(
                bollard::errors::Error::DockerResponseServerError {
                    status_code: 404, ..
                },
            )) => {
                self.submitted = false; // A joined image-not-found reply is definitive.
                publish_submission(false);
                client
                    .pull_image(
                        &container_req.descriptor(),
                        container_req.platform().clone(),
                    )
                    .await?;
                self.submitted = true;
                publish_submission(true);
                client.create_container(create_options, config).await
            }
            other => other,
        };
        // This boundary sees only genuine POST replies, never a later Image callback error.
        self.collision = create_collision(&final_result);
        if self.collision
            || matches!(
                &final_result,
                Err(ClientError::CreateContainer(
                    bollard::errors::Error::DockerResponseServerError {
                        status_code: 404,
                        ..
                    }
                ))
            )
        {
            self.submitted = false; // Joined409 or image404 certifies no creation by this POST.
        }
        publish_submission(self.submitted);
        let container_id = final_result?;
        self.id = Some(container_id.clone());
        self.publish();
        Ok(container_id)
    }
}

fn create_collision(result: &std::result::Result<String, ClientError>) -> bool {
    matches!(
        result,
        Err(ClientError::CreateContainer(
            bollard::errors::Error::DockerResponseServerError {
                status_code: 409,
                ..
            }
        ))
    )
}

impl From<&Mount> for bollard::models::Mount {
    fn from(mount: &Mount) -> Self {
        let mount_type = match mount.mount_type() {
            MountType::Bind => bollard::models::MountTypeEnum::BIND,
            MountType::Volume => bollard::models::MountTypeEnum::VOLUME,
            MountType::Tmpfs => bollard::models::MountTypeEnum::TMPFS,
        };

        let is_read_only = matches!(mount.access_mode(), AccessMode::ReadOnly);

        Self {
            target: mount.target().map(str::to_string),
            source: mount.source().map(str::to_string),
            typ: Some(mount_type),
            read_only: Some(is_read_only),
            tmpfs_options: mount
                .tmpfs_options()
                .map(|opts| bollard::models::MountTmpfsOptions {
                    size_bytes: opts.size_bytes,
                    mode: opts.mode,
                    options: None,
                }),
            ..Default::default()
        }
    }
}

impl From<CgroupnsMode> for HostConfigCgroupnsModeEnum {
    fn from(value: CgroupnsMode) -> Self {
        match value {
            CgroupnsMode::Host => Self::HOST,
            CgroupnsMode::Private => Self::PRIVATE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::{BuildImageOptions, IntoContainerPort, WaitFor},
        images::generic::GenericImage,
        runners::AsyncBuilder,
        GenericBuildableImage, ImageExt,
    };

    async fn get_server_image() -> GenericImage {
        let generic_image = GenericBuildableImage::new("simple_web_server", "latest")
            // "Dockerfile" is included already, so adding the build context directory is all what is needed
            .with_file(
                std::fs::canonicalize("../testimages/simple_web_server").unwrap(),
                ".",
            )
            .build_image_with(BuildImageOptions::new())
            .await
            .unwrap();
        generic_image.with_wait_for(WaitFor::message_on_stdout("server is ready"))
    }

    /// Test that all user-supplied labels are added to containers started by `AsyncRunner::start`
    #[tokio::test]
    async fn async_start_should_apply_expected_labels() -> anyhow::Result<()> {
        let mut labels = HashMap::from([
            ("foo".to_string(), "bar".to_string()),
            ("baz".to_string(), "qux".to_string()),
            // include a `managed-by` value to guard against future changes
            // inadvertently allowing users to override keys we rely on to
            // internally ensure sane and correct behavior
            (
                "org.testcontainers.managed-by".to_string(),
                "the-time-wizard".to_string(),
            ),
        ]);

        let image = GenericImage::new("testcontainers/helloworld", "1.3.0").with_labels(&labels);

        let container = {
            #[cfg(not(feature = "reusable-containers"))]
            {
                image
            }
            #[cfg(feature = "reusable-containers")]
            {
                image.with_reuse(crate::ReuseDirective::CurrentSession)
            }
        }
        .start()
        .await?;

        let client = Client::lazy_client().await?;

        let container_labels = client
            .inspect(container.id())
            .await?
            .config
            .unwrap_or_default()
            .labels
            .unwrap_or_default();

        // the created labels and container labels shouldn't actually be identical, as the
        // `org.testcontainers.managed-by: testcontainers` label is always unconditionally
        // applied to all containers by `AsyncRunner::start`, with the value `testcontainers`
        // being applied *last* explicitly so that even user-supplied values of the
        // `org.testcontainers.managed-by` key will be overwritten
        assert_ne!(&labels, &container_labels);

        // If we add the expected `managed-by` value though, they should then match
        labels.insert(
            "org.testcontainers.managed-by".to_string(),
            "testcontainers".to_string(),
        );

        #[cfg(feature = "reusable-containers")]
        labels.extend([(
            "org.testcontainers.session-id".to_string(),
            session_id().to_string(),
        )]);

        assert_eq!(labels, container_labels);

        container.rm().await.map_err(anyhow::Error::from)
    }

    #[tokio::test]
    async fn async_run_command_should_expose_all_ports_if_no_explicit_mapping_requested(
    ) -> anyhow::Result<()> {
        let client = Client::lazy_client().await?;
        let container = GenericImage::new("testcontainers/helloworld", "1.3.0")
            .start()
            .await?;

        let container_details = client.inspect(container.id()).await?;
        let publish_ports = container_details
            .host_config
            .expect("HostConfig")
            .publish_all_ports
            .expect("PublishAllPorts");
        assert!(publish_ports, "publish_all_ports must be `true`");
        Ok(())
    }

    #[tokio::test]
    async fn async_run_command_should_map_exposed_port() -> anyhow::Result<()> {
        let _ = pretty_env_logger::try_init();
        let image = get_server_image().await.with_exposed_port(5000.tcp());
        let container = image.start().await?;
        container
            .get_host_port_ipv4(5000.tcp())
            .await
            .expect("Port should be mapped");
        Ok(())
    }

    #[tokio::test]
    #[cfg(feature = "device-requests")]
    async fn async_run_command_should_request_gpu() -> anyhow::Result<()> {
        use anyhow::Context as _;
        use tokio::io::AsyncReadExt as _;

        let _ = pretty_env_logger::try_init();

        // PRECONDITION: the host has GPU and NVIDIA drivers
        let host_has_gpu = std::process::Command::new("nvidia-smi")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);
        if !host_has_gpu {
            log::warn!("Host does not have GPU or NVIDIA drivers, context: https://www.nvidia.com/en-us/drivers/");
            return Ok(());
        }

        // PRECONDITION: Docker on host has NVIDIA runtime
        let client = bollard::Docker::connect_with_defaults()?;
        let info = client.info().await?;
        let has_nvidia_runtimes = info
            .runtimes
            .context("No runtimes found")?
            .iter()
            .any(|(key, _runtime)| key.contains("nvidia"));
        if !has_nvidia_runtimes {
            log::warn!("No Docker NVIDIA runtime installed, context: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html");
            return Ok(());
        }

        // GIVEN: a container with a device request for GPU
        let device_request = bollard::models::DeviceRequest {
            driver: Some(String::from("nvidia")),
            count: Some(-1), // expose all
            capabilities: Some(vec![vec![String::from("gpu")]]),
            device_ids: None,
            options: None,
        };

        // The image must have nvidia drivers installed, so we don't use the simple server here
        let image = GenericImage::new("ubuntu", "24.04")
            .with_device_requests(vec![device_request])
            .with_cmd(["sleep", "infinity"]); // keep the container running to have time to run the command
        let container = image.start().await?;

        // WHEN: `nvidia-smi` command is executed
        let mut result = container
            .exec(crate::core::ExecCommand::new(["nvidia-smi"]))
            .await?;

        // THEN: the output contains the expected "NVIDIA-SMI" string
        let mut output: String = String::new();
        let mut stdout = result.stdout();
        stdout.read_to_string(&mut output).await?;

        assert!(
            output.contains("NVIDIA-SMI"),
            "Could not access NVIDIA System Management Interface"
        );

        Ok(())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn async_run_command_should_map_exposed_port_udp_sctp() -> anyhow::Result<()> {
        let client = Client::lazy_client().await?;
        let _ = pretty_env_logger::try_init();

        let udp_port = 1000.udp();
        let sctp_port = 2000.sctp();

        let generic_server = get_server_image()
            .await
            // Explicitly expose the port, which otherwise would not be available.
            .with_exposed_port(udp_port)
            .with_exposed_port(sctp_port);

        let container = generic_server.start().await?;

        let container_details = client.inspect(container.id()).await?;

        let current_ports_map = container_details
            .network_settings
            .expect("network_settings")
            .ports
            .expect("ports");

        let mut current_ports = current_ports_map.keys().collect::<Vec<&String>>();

        current_ports.sort();

        let mut expected_ports: Vec<&String> = Vec::new();

        let tcp_expected_port = &String::from("80/tcp");
        let sctp_expected_port = &String::from("2000/sctp");
        let udp_expected_port = &String::from("1000/udp");

        expected_ports.push(udp_expected_port);
        expected_ports.push(sctp_expected_port);
        expected_ports.push(tcp_expected_port);

        assert!(
            expected_ports
                .iter()
                .all(|port| current_ports.contains(port)),
            "exposed ports: {current_ports:?} doesn't contain all expected ports: {expected_ports:?}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn async_run_command_should_expose_only_requested_ports() -> anyhow::Result<()> {
        let client = Client::lazy_client().await?;
        let image = GenericImage::new("testcontainers/helloworld", "1.3.0");
        let container = image
            .with_mapped_port(123, 456.tcp())
            .with_mapped_port(555, 888.tcp())
            .start()
            .await?;

        let container_details = client.inspect(container.id()).await?;

        let port_bindings = container_details
            .host_config
            .expect("HostConfig")
            .port_bindings
            .expect("PortBindings");
        assert!(
            port_bindings.contains_key("456/tcp"),
            "port 456/tcp must be mapped"
        );
        assert!(
            port_bindings.contains_key("888/tcp"),
            "port 888/tcp must be mapped"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn async_run_command_should_map_ports_udp_sctp() -> anyhow::Result<()> {
        let client = Client::lazy_client().await?;
        let _ = pretty_env_logger::try_init();

        let udp_port = 1000;
        let sctp_port = 2000;

        let image = GenericImage::new("testcontainers/helloworld", "1.3.0");
        let container = image
            .with_mapped_port(123, udp_port.udp())
            .with_mapped_port(555, sctp_port.sctp())
            .start()
            .await?;

        let container_details = client.inspect(container.id()).await?;

        let current_ports_map = container_details
            .host_config
            .expect("HostConfig")
            .port_bindings
            .expect("ports");

        let mut current_ports = current_ports_map.keys().collect::<Vec<&String>>();

        current_ports.sort();

        let mut expected_ports: Vec<&String> = Vec::new();

        let sctp_expected_port = &String::from("2000/sctp");
        let udp_expected_port = &String::from("1000/udp");

        expected_ports.push(udp_expected_port);
        expected_ports.push(sctp_expected_port);

        assert_eq!(current_ports, expected_ports);

        Ok(())
    }

    #[tokio::test]
    async fn async_run_command_should_include_network() -> anyhow::Result<()> {
        let client = Client::lazy_client().await?;
        let image = GenericImage::new("testcontainers/helloworld", "1.3.0");
        let container = image.with_network("awesome-net-1").start().await?;

        let container_details = client.inspect(container.id()).await?;
        let networks = container_details
            .network_settings
            .expect("NetworkSettings")
            .networks
            .expect("Networks");

        assert!(
            networks.contains_key("awesome-net-1"),
            "Networks is {networks:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn async_run_command_should_include_name() -> anyhow::Result<()> {
        let client = Client::lazy_client().await?;
        let image = GenericImage::new("testcontainers/helloworld", "1.3.0");
        let container = image
            .with_container_name("async_hello_container")
            .start()
            .await?;

        let container_details = client.inspect(container.id()).await?;
        let container_name = container_details.name.expect("Name");
        assert!(container_name.ends_with("async_hello_container"));
        Ok(())
    }

    #[tokio::test]
    async fn async_should_create_network_if_image_needs_it_and_drop_it_in_the_end(
    ) -> anyhow::Result<()> {
        let hello_world = GenericImage::new("testcontainers/helloworld", "1.3.0");

        {
            let client = Client::lazy_client().await?;
            assert!(!client.network_exists("awesome-net-2").await?);

            // creating the first container creates the network
            let _container1 = hello_world
                .clone()
                .with_network("awesome-net-2")
                .start()
                .await;

            // creating a 2nd container doesn't fail because check if the network exists already
            let _container2 = hello_world.with_network("awesome-net-2").start().await;

            assert!(client.network_exists("awesome-net-2").await?);
        }

        // containers have been dropped, should clean up networks
        tokio::time::sleep(Duration::from_secs(1)).await;
        let client = Client::lazy_client().await?;
        assert!(!client.network_exists("awesome-net-2").await?);
        Ok(())
    }

    #[tokio::test]
    async fn async_should_rely_on_network_mode_when_network_is_provided_and_settings_bridge_empty(
    ) -> anyhow::Result<()> {
        let web_server = get_server_image().await.with_wait_for(WaitFor::seconds(1));

        let container = web_server.clone().with_network("bridge").start().await?;

        assert!(
            !container
                .get_bridge_ip_address()
                .await?
                .to_string()
                .is_empty(),
            "Bridge IP address must not be empty"
        );
        Ok(())
    }

    #[tokio::test]
    async fn async_should_return_error_when_non_bridged_network_selected() -> anyhow::Result<()> {
        let web_server = get_server_image().await.with_wait_for(WaitFor::seconds(1));

        let container = web_server.clone().with_network("host").start().await?;

        let res = container.get_bridge_ip_address().await;
        assert!(
            res.is_err(),
            "Getting bridge IP address should fail due to network mode"
        );
        Ok(())
    }

    #[tokio::test]
    async fn async_run_command_should_set_shared_memory_size() -> anyhow::Result<()> {
        let client = Client::lazy_client().await?;
        let image = GenericImage::new("testcontainers/helloworld", "1.3.0");
        let container = image.with_shm_size(1_000_000).start().await?;

        let container_details = client.inspect(container.id()).await?;
        let shm_size = container_details
            .host_config
            .expect("HostConfig")
            .shm_size
            .expect("ShmSize");

        assert_eq!(shm_size, 1_000_000);
        Ok(())
    }

    #[tokio::test]
    async fn async_run_command_should_include_privileged() -> anyhow::Result<()> {
        let image = GenericImage::new("testcontainers/helloworld", "1.3.0");
        let container = image.with_privileged(true).start().await?;

        let client = Client::lazy_client().await?;
        let container_details = client.inspect(container.id()).await?;

        let privileged = container_details
            .host_config
            .expect("HostConfig")
            .privileged
            .expect("Privileged");
        assert!(privileged, "privileged must be `true`");
        Ok(())
    }

    #[tokio::test]
    async fn async_run_command_should_have_cap_add() -> anyhow::Result<()> {
        let image = GenericImage::new("testcontainers/helloworld", "1.3.0");
        let expected_capability = "NET_ADMIN";
        let container = image
            .with_cap_add(expected_capability.to_string())
            .start()
            .await?;

        let client = Client::lazy_client().await?;
        let container_details = client.inspect(container.id()).await?;

        let capabilities = container_details
            .host_config
            .expect("HostConfig")
            .cap_add
            .expect("CapAdd");

        assert_eq!(
            expected_capability,
            capabilities.first().expect("No capabilities added"),
            "cap_add must contain {expected_capability}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn async_run_command_should_have_cap_drop() -> anyhow::Result<()> {
        let image = GenericImage::new("testcontainers/helloworld", "1.3.0");
        let expected_capability = "AUDIT_WRITE";
        let container = image
            .with_cap_drop(expected_capability.to_string())
            .start()
            .await?;

        let client = Client::lazy_client().await?;
        let container_details = client.inspect(container.id()).await?;

        let capabilities = container_details
            .host_config
            .expect("HostConfig")
            .cap_drop
            .expect("CapAdd");

        assert_eq!(
            expected_capability,
            capabilities.first().expect("No capabilities dropped"),
            "cap_drop must contain {expected_capability}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn async_run_command_should_include_ulimits() -> anyhow::Result<()> {
        let image = GenericImage::new("testcontainers/helloworld", "1.3.0");
        let container = image.with_ulimit("nofile", 123, Some(456)).start().await?;

        let client = Client::lazy_client().await?;
        let container_details = client.inspect(container.id()).await?;

        let ulimits = container_details
            .host_config
            .expect("HostConfig")
            .ulimits
            .expect("Privileged");

        assert_eq!(ulimits.len(), 1);
        assert_eq!(ulimits[0].name, Some("nofile".into()));
        assert_eq!(ulimits[0].soft, Some(123));
        assert_eq!(ulimits[0].hard, Some(456));
        Ok(())
    }

    #[tokio::test]
    async fn async_run_command_should_have_host_cgroupns_mode() -> anyhow::Result<()> {
        let image = GenericImage::new("testcontainers/helloworld", "1.3.0");
        let container = image.with_cgroupns_mode(CgroupnsMode::Host).start().await?;

        let client = Client::lazy_client().await?;
        let container_details = client.inspect(container.id()).await?;

        let cgroupns_mode = container_details
            .host_config
            .expect("HostConfig")
            .cgroupns_mode
            .expect("CgroupnsMode");

        assert_eq!(
            HostConfigCgroupnsModeEnum::HOST,
            cgroupns_mode,
            "cgroupns mode must be `host`"
        );
        Ok(())
    }

    #[tokio::test]
    async fn async_run_command_should_have_private_cgroupns_mode() -> anyhow::Result<()> {
        let image = GenericImage::new("testcontainers/helloworld", "1.3.0");
        let container = image
            .with_cgroupns_mode(CgroupnsMode::Private)
            .start()
            .await?;

        let client = Client::lazy_client().await?;
        let container_details = client.inspect(container.id()).await?;

        let cgroupns_mode = container_details
            .host_config
            .expect("HostConfig")
            .cgroupns_mode
            .expect("CgroupnsMode");

        assert_eq!(
            HostConfigCgroupnsModeEnum::PRIVATE,
            cgroupns_mode,
            "cgroupns mode must be `private`"
        );
        Ok(())
    }

    #[tokio::test]
    async fn async_run_command_should_have_host_userns_mode() -> anyhow::Result<()> {
        let image = GenericImage::new("testcontainers/helloworld", "1.3.0");
        let container = image.with_userns_mode("host").start().await?;

        let client = Client::lazy_client().await?;
        let container_details = client.inspect(container.id()).await?;

        let userns_mode = container_details
            .host_config
            .expect("HostConfig")
            .userns_mode
            .expect("UsernsMode");
        assert_eq!("host", userns_mode, "userns mode must be `host`");
        Ok(())
    }

    #[tokio::test]
    async fn async_run_command_should_have_working_dir() -> anyhow::Result<()> {
        let image = GenericImage::new("testcontainers/helloworld", "1.2.0");
        let expected_working_dir = "/foo";
        let container = image.with_working_dir(expected_working_dir).start().await?;

        let client = Client::lazy_client().await?;
        let container_details = client.inspect(container.id()).await?;

        let working_dir = container_details
            .config
            .expect("ContainerConfig")
            .working_dir
            .expect("WorkingDir");
        assert_eq!(
            expected_working_dir, &working_dir,
            "working dir must be `foo`"
        );
        Ok(())
    }

    #[tokio::test]
    async fn async_run_command_should_have_user() -> anyhow::Result<()> {
        let image = get_server_image().await;
        let expected_user = "root";
        let container = image.with_user(expected_user).start().await?;

        let client = Client::lazy_client().await?;
        let container_details = client.inspect(container.id()).await?;

        let user = container_details
            .config
            .expect("ContainerConfig")
            .user
            .expect("User");
        assert_eq!(expected_user, &user, "user must be `root`");
        Ok(())
    }

    #[cfg(feature = "reusable-containers")]
    #[tokio::test]
    async fn async_start_should_reuse_stopped_container() -> anyhow::Result<()> {
        use crate::ReuseDirective;

        let client = Client::lazy_client().await?;
        let container_name = format!("test-reuse-stopped-{}", std::process::id());

        let container1 = GenericImage::new("testcontainers/helloworld", "1.3.0")
            .with_container_name(&container_name)
            .with_reuse(ReuseDirective::Always)
            .start()
            .await?;
        let container_id_1 = container1.id().to_string();

        assert!(
            client.container_is_running(&container_id_1).await?,
            "Container should be running after start"
        );

        container1.stop().await?;
        assert!(
            !client.container_is_running(&container_id_1).await?,
            "Container should be stopped after stop()"
        );

        let container2 = GenericImage::new("testcontainers/helloworld", "1.3.0")
            .with_container_name(&container_name)
            .with_reuse(ReuseDirective::Always)
            .start()
            .await?;
        let container_id_2 = container2.id().to_string();

        assert_eq!(
            container_id_1, container_id_2,
            "Should reuse the same container ID"
        );
        assert!(
            client.container_is_running(&container_id_2).await?,
            "Container should be running after restart"
        );
        container2.rm().await.map_err(anyhow::Error::from)
    }
}

#[cfg(test)]
mod retained_api_tests {
    use super::*;
    use crate::{
        core::{ExecCommand, WaitFor},
        GenericImage, ImageExt,
    };
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    fn post_error(status_code: u16) -> ClientError {
        ClientError::CreateContainer(bollard::errors::Error::DockerResponseServerError {
            status_code,
            message: "regression".into(),
        })
    }

    #[test]
    fn retained_run_is_send_without_polling_or_starting_a_runtime() {
        fn require_send<F: std::future::Future + Send>(_: F) {}
        let generic = StartupOwner::new(GenericImage::new("alpine", "3.21").into());
        require_send(generic.run());
        let callback = StartupOwner::new(
            CallbackImage {
                mode: 2,
                readiness_calls: Arc::new(AtomicUsize::new(0)),
            }
            .into(),
        );
        require_send(callback.run());
    }

    #[test]
    fn collision_is_post_provenance_not_callback_error_shape() {
        assert!(create_collision(&Err(post_error(409))));
        assert!(!create_collision(&Err(post_error(404))));
        assert!(!create_collision(&Err(post_error(500))));
        assert!(!create_collision(&Ok("acknowledged-id".into())));
        let mut state = StartupState::new(GenericImage::new("alpine", "3.21").into());
        state.fail(post_error(409).into()); // Same error may legally come from an Image callback.
        assert!(!state.collision);
        assert!(state.failure.is_some());
    }

    async fn diagnostic_exclusion() {
        let owner = StartupOwner::new(GenericImage::new("alpine", "3.21").into());
        let alias = owner.clone();
        let guard = owner.0.lock().await;
        let before = alias.snapshot(); // Must finish while the startup mutex is held.
        assert!(!before.name.is_empty());
        assert!(!before.owner_token.is_empty());
        assert!(alias.run().await.is_err());
        assert!(!alias.close_retained().await.unwrap());
        assert!(alias.take_ready().await.is_err());
        assert_eq!(alias.snapshot().owner_token, before.owner_token);
        drop(guard);
        assert!(owner.close_retained().await.unwrap()); // Component only: no POST admitted.
        assert!(owner.close_retained().await.unwrap());
        assert_eq!(owner.snapshot().phase, "retired");
        let collision = StartupOwner::new(GenericImage::new("alpine", "3.21").into());
        {
            let mut state = collision.0.lock().await;
            state.started = true;
            state.collision = true;
            state.fail(post_error(409).into());
            assert!(!state.submitted); // Component: collected definitive no-create, no client.
        }
        assert!(collision.close_retained().await.unwrap());
        assert!(collision.snapshot().failure.is_some());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn diagnostics_during_locked_startup_current_thread() {
        diagnostic_exclusion().await;
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn diagnostics_during_locked_startup_multi_thread() {
        diagnostic_exclusion().await;
    }

    #[test]
    fn recovery_identity_errors_latch_original_failure() {
        let owner = StartupOwner::new(GenericImage::new("alpine", "3.21").into());
        let mut state = owner.0.try_lock().unwrap();
        let valid = bollard::models::ContainerInspectResponse {
            id: Some("id".into()),
            name: Some(format!("/{}", state.name)),
            config: Some(bollard::models::ContainerConfig {
                labels: Some(HashMap::from([(OWNER_LABEL.into(), state.token.clone())])),
                ..Default::default()
            }),
            ..Default::default()
        };
        for case in 0..5 {
            let mut invalid = valid.clone();
            match case {
                0 => invalid.name = Some("/foreign".into()),
                1 => invalid.config = None,
                2 => {
                    invalid.config.as_mut().unwrap().labels = Some(HashMap::from([(
                        OWNER_LABEL.into(),
                        "foreign-token".into(),
                    )]))
                }
                3 => invalid.id = None,
                _ => invalid.id = Some(String::new()),
            }
            let error = state.recovered_id(invalid).unwrap_err();
            state.fail(error);
            assert!(owner.snapshot().failure.is_some());
        }
        assert_eq!(state.recovered_id(valid).unwrap(), "id");
        assert!(state.failure.is_some()); // Later valid evidence never erases the error.
    }

    #[derive(Debug)]
    struct CallbackImage {
        mode: u8,
        readiness_calls: Arc<AtomicUsize>,
    }
    impl Image for CallbackImage {
        fn name(&self) -> &str {
            "alpine"
        }
        fn tag(&self) -> &str {
            "3.21"
        }
        fn entrypoint(&self) -> Option<&str> {
            (self.mode == 1).then_some("/testcontainers-missing-entrypoint")
        }
        fn ready_conditions(&self) -> Vec<WaitFor> {
            self.readiness_calls.fetch_add(1, Ordering::SeqCst);
            vec![]
        }
        fn exec_before_ready(&self, _: ContainerState) -> Result<Vec<ExecCommand>> {
            if self.mode == 2 {
                Err(post_error(409).into())
            } else {
                Ok(vec![])
            }
        }
        fn exec_after_start(&self, _: ContainerState) -> Result<Vec<ExecCommand>> {
            if self.mode == 3 {
                Err(post_error(409).into())
            } else {
                Ok(vec![])
            }
        }
    }

    async fn real_callback_regressions() {
        for mode in 0..4 {
            let calls = Arc::new(AtomicUsize::new(0));
            let mut request = CallbackImage {
                mode,
                readiness_calls: calls.clone(),
            }
            .with_cmd(["sleep", "60"]);
            if mode == 0 {
                request = request.with_copy_to(
                    "/fixture",
                    std::path::PathBuf::from(format!(
                        "/testcontainers-missing-{}",
                        uuid::Uuid::new_v4()
                    )),
                );
            }
            let owner = StartupOwner::new(request);
            let task_owner = owner.clone();
            let joined = tokio::spawn(async move { task_owner.run().await })
                .await
                .unwrap();
            assert!(joined.is_err());
            let (id, client, address) = {
                let state = owner.0.lock().await;
                let actual = state.owner.as_ref().expect("actual ACK owner retained");
                assert_eq!(actual.log_consumers_started, mode >= 2);
                assert_eq!(calls.load(Ordering::SeqCst), usize::from(mode >= 2));
                assert!(!state.collision, "callback409 is not a POST collision");
                (
                    actual.id().to_string(),
                    actual.docker_client().clone(),
                    actual as *const _ as usize,
                )
            };
            let failure = owner.snapshot().failure.expect("joined failure latched");
            assert_eq!(owner.snapshot().id.as_deref(), Some(id.as_str()));
            assert!(owner.run().await.is_err()); // Duplicate never replaces the stored owner.
            {
                let state = owner.0.lock().await;
                assert_eq!(state.owner.as_ref().unwrap() as *const _ as usize, address);
            }
            assert!(owner.close_retained().await.unwrap());
            assert!(owner.close_retained().await.unwrap());
            assert_eq!(owner.snapshot().failure.as_deref(), Some(failure.as_str()));
            assert!(!owner.snapshot().owner_present);
            assert!(matches!(
                client.inspect(&id).await,
                Err(ClientError::InspectContainer(
                    bollard::errors::Error::DockerResponseServerError {
                        status_code: 404,
                        ..
                    }
                ))
            ));
            assert!(owner.take_ready().await.is_err());
        }
    }

    #[tokio::test(flavor = "current_thread")]
    #[ignore = "requires real Engine; main runs explicitly after source handoff"]
    async fn retained_callback_order_and_cleanup_current_thread() {
        real_callback_regressions().await;
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "requires real Engine; main runs explicitly after source handoff"]
    async fn retained_callback_order_and_cleanup_multi_thread() {
        real_callback_regressions().await;
    }
}
