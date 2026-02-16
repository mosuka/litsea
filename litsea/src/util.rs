use std::fmt;
use std::str::FromStr;

/// URI scheme for loading models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelScheme {
    /// HTTP scheme for remote model loading.
    Http,
    /// HTTPS scheme for secure remote model loading.
    Https,
    /// Local file system scheme.
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

impl fmt::Display for ModelScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_str_valid() {
        assert!(matches!("http".parse::<ModelScheme>(), Ok(ModelScheme::Http)));
        assert!(matches!("https".parse::<ModelScheme>(), Ok(ModelScheme::Https)));
        assert!(matches!("file".parse::<ModelScheme>(), Ok(ModelScheme::File)));
    }

    #[test]
    fn test_from_str_invalid() {
        assert!("ftp".parse::<ModelScheme>().is_err());
        assert!("".parse::<ModelScheme>().is_err());
    }

    #[test]
    fn test_as_str_roundtrip() {
        for scheme_str in &["http", "https", "file"] {
            let scheme: ModelScheme = scheme_str.parse().unwrap();
            assert_eq!(scheme.as_str(), *scheme_str);
        }
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", ModelScheme::Http), "http");
        assert_eq!(format!("{}", ModelScheme::Https), "https");
        assert_eq!(format!("{}", ModelScheme::File), "file");
    }
}
