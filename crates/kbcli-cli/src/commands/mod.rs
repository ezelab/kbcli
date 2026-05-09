//! Top-level dispatch for `kbcli` subcommands.

use clap::Subcommand;

use kbcli_core::Result;

mod db;
mod doc;
mod query;

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Manage local databases.
    #[command(subcommand)]
    Db(db::DbCommand),

    /// Manage documents inside a database.
    #[command(subcommand)]
    Doc(doc::DocCommand),

    /// Query a database.
    Query(query::QueryArgs),
}

pub async fn dispatch(cmd: Command, json: bool) -> Result<()> {
    match cmd {
        Command::Db(c) => db::run(c, json).await,
        Command::Doc(c) => doc::run(c, json).await,
        Command::Query(a) => query::run(a, json).await,
    }
}
