use aien_local_stack::LocalStack;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Initializing AIEN Local Stack (In-Process Sovereign Composition)...");
    let _stack = LocalStack::new()?;
    info!("Local stack initialized successfully. Inference and agent plane unified.");

    Ok(())
}
