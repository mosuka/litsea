use std::str::FromStr;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum ModelScheme {
    #[serde(rename = "http")]
    Http,
    #[serde(rename = "https")]
    Https,
    #[serde(rename = "file")]
    File,
}

impl ModelScheme {
    pub fn as_str(&self) -> &str {
        match self {
            ModelScheme::Http => "http",
            ModelScheme::Https => "https",
            ModelScheme::File => "file",
        }
    }
}

impl FromStr for ModelScheme {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "http" => Ok(ModelScheme::Http),
            "https" => Ok(ModelScheme::Https),
            "file" => Ok(ModelScheme::File),
            _ => Err(format!("Invalid model scheme: {}", s)),
        }
    }
}
