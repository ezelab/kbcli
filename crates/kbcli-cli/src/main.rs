//! kbcli command-line interface.

mod commands;
mod io;
mod paths;
mod pipeline;
mod runtime_factory;
mod store_factory;

use std::process::ExitCode;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "kbcli",
    version,
    about = "Fully-local CLI for creating and querying semantic databases of documents.",
    long_about = None,
)]
struct Cli {
    /// Increase log verbosity (-v info, -vv debug, -vvv trace).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Render output as machine-readable JSON instead of human-friendly text.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: commands::Command,
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    match commands::dispatch(cli.command, cli.json).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            if cli.json {
                let payload = serde_json::json!({
                    "ok": false,
                    "error": {
                        "code": e.code(),
                        "message": e.to_string(),
                    }
                });
                println!("{}", payload);
            } else {
                eprintln!("error: {e}");
            }
            ExitCode::from(1)
        }
    }
}

fn init_tracing(verbosity: u8) {
    use tracing_subscriber::{fmt, EnvFilter};

    let default_level = match verbosity {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("kbcli={default_level},warn")));

    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .compact()
        .try_init()
        .ok();
}
